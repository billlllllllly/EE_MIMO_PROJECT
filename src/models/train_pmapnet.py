"""Train PMAP-Net on saved z, R, and local bit LLRs from Partial MAP.

Run from the project root as python -m src.models.train_pmapnet. Mount
Google Drive in the Colab notebook before running with the default output path.
"""

import argparse
import csv
import io
import math
import zipfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import mse_loss

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .PMAPnet import PMAPNet
else:
    from PMAPnet import PMAPNet


DEFAULT_OUTPUT_DIR = Path("/content/drive/MyDrive/EE_MIMO_PROJECT/02_colab_output/")


def constellation_for(modulation: str) -> tuple[Tensor, Tensor]:
    if modulation == "bpsk":
        points = [-1, 1]
        labels = [[0], [1]]
    else:
        points = [-1 - 1j, -1 + 1j, 1 + 1j, 1 - 1j]
        labels = [[0, 0], [1, 0], [1, 1], [0, 1]]
    # Match the constellation and label order used in rayleigh.ipynb.
    constellation = torch.tensor(points, dtype=torch.complex128) / math.sqrt(2)
    bit_labels = torch.tensor(labels, dtype=torch.long)
    return constellation, bit_labels


def pack_features(z: Tensor, r_row: Tensor, constellation: Tensor) -> Tensor:
    """Build Eq. (32) features with all real parts followed by all imaginary parts."""
    products = (r_row.unsqueeze(-1) * constellation).flatten(start_dim=-2)
    products = products.expand(*z.shape, products.shape[-1])
    values = torch.cat((z.unsqueeze(-1), products), dim=-1)
    return torch.cat((values.real, values.imag), dim=-1).float()


def resolve_layer_data(path: Path, num_tx: int) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Could not find saved Partial MAP training data at {path}.")

    layer_files = [f"layer_{row}.csv" for row in range(num_tx)]
    missing = [name for name in layer_files if not (path / name).is_file()]
    if not missing:
        return path

    archive = path / "rayleigh_pmap_layers.zip"
    if archive.is_file():
        return archive
    raise FileNotFoundError(
        f"Missing {', '.join(missing)} in {path}. "
        "Provide the layer CSV files or rayleigh_pmap_layers.zip."
    )


def load_layer_data(
    path: Path,
    num_tx: int,
    constellation: Tensor,
    sample_count: int,
    seed: int,
    device: torch.device,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Read the same randomly selected records from every Partial MAP layer."""
    bits_per_symbol = len(constellation).bit_length() - 1
    constellation = constellation.to(device)

    with ExitStack() as stack:
        archive = None if path.is_dir() else stack.enter_context(zipfile.ZipFile(path))

        def open_layer(layer: int):
            name = f"layer_{layer}.csv"
            if archive is None:
                return (path / name).open(encoding="utf-8", newline="")
            if name not in archive.namelist():
                raise ValueError(f"{path} is missing {name} at the archive root.")
            return io.TextIOWrapper(archive.open(name), encoding="utf-8", newline="")

        with open_layer(0) as stream:
            next(stream, None)
            row_count = sum(1 for _ in stream)
        if row_count < sample_count:
            raise ValueError(
                f"layer_0.csv contains {row_count:,} data rows. "
                f"At least {sample_count:,} are required."
            )

        indices = np.random.default_rng(seed).choice(
            row_count, size=sample_count, replace=False
        )
        indices.sort()
        features_by_row = []
        targets_by_row = []
        for layer in range(num_tx):
            remaining = num_tx - layer
            input_columns = [f"z{layer}_re", f"z{layer}_im"]
            for tx in range(layer, num_tx):
                input_columns.extend([f"r{layer}{tx}_re", f"r{layer}{tx}_im"])
            target_columns = [
                f"llr_tx{tx}_bit{bit}"
                for tx in range(layer, num_tx)
                for bit in range(bits_per_symbol)
            ]
            header = input_columns + target_columns
            name = f"layer_{layer}.csv"
            values = np.empty((sample_count, len(header)), dtype=np.float32)

            selected = 0
            seen = 0
            with open_layer(layer) as stream:
                reader = csv.reader(stream, strict=True)
                if next(reader, None) != header:
                    raise ValueError(f"{name} has an unexpected Partial MAP header.")
                for seen, fields in enumerate(reader, start=1):
                    if len(fields) != len(header):
                        raise ValueError(f"{name} has an invalid data row {seen}.")
                    if selected < sample_count and seen - 1 == indices[selected]:
                        try:
                            values[selected] = fields
                        except ValueError as exc:
                            raise ValueError(
                                f"{name} has a non-numeric value in data row {seen}."
                            ) from exc
                        selected += 1

            if seen != row_count or selected != sample_count:
                raise ValueError(
                    f"{name} has {seen:,} data rows but layer_0.csv has {row_count:,}."
                )
            if not np.isfinite(values).all():
                raise ValueError(f"{name} has a non-finite feature or LLR.")

            input_count = len(input_columns)
            pairs = torch.from_numpy(values[:, :input_count]).reshape(
                sample_count, remaining + 1, 2
            )
            complex_inputs = torch.complex(pairs[..., 0], pairs[..., 1]).to(device)
            features = pack_features(
                complex_inputs[:, 0], complex_inputs[:, 1:], constellation
            )
            targets = torch.from_numpy(values[:, input_count:]).to(device)
            features_by_row.append(features)
            targets_by_row.append(targets)

    return tuple(features_by_row), tuple(targets_by_row)


def fit(model, features, targets, args, generator):
    """Fit the independent QR-row predictors on sample-weighted MSE."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # The paper's implementation section gives this exponential decay.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 0.97 ** (step / 500)
    )
    device = features[0].device
    sample_count = len(features[0])
    history = []
    for epoch in range(1, args.epochs + 1):
        totals = torch.zeros(args.num_tx, device=device)
        shuffled = torch.randperm(sample_count, device=device, generator=generator)
        for indices in shuffled.split(args.batch_size):
            predictions = model(tuple(field[indices] for field in features))
            losses = torch.stack([
                mse_loss(prediction, target[indices])
                for prediction, target in zip(predictions, targets)
            ])
            optimizer.zero_grad(set_to_none=True)
            losses.sum().backward()
            optimizer.step()
            scheduler.step()
            totals.add_(losses.detach(), alpha=len(indices))

        row_losses = (totals / sample_count).tolist()
        if not all(map(math.isfinite, row_losses)):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
        history.append(row_losses)
        summary = ", ".join(f"row {row} {loss:.5g}" for row, loss in enumerate(row_losses, 1))
        print(f"Epoch {epoch:3}/{args.epochs}: {summary}", flush=True)
    return history


def save_loss_plot(history, modulation, path):
    figure, axis = plt.subplots(figsize=(6.5, 4.4), layout="constrained")
    for row, losses in enumerate(zip(*history), 1):
        axis.plot(
            range(1, len(losses) + 1), losses,
            label=f"QR row {row}", marker="o" if len(losses) == 1 else None,
        )
    axis.set(title=modulation.upper(), xlabel="Epoch", ylabel="Training MSE")
    axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside lower center", ncol=min(4, len(labels)), frameon=False
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulation", choices=("bpsk", "qpsk"), default="bpsk")
    parser.add_argument("--num-tx", type=int, default=4)
    parser.add_argument(
        "--hidden-layers", type=int, default=2, help="Depth used in Eq. (36)'s example."
    )
    parser.add_argument("--train-samples", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--layer-data", type=Path, required=True,
        help="Directory with layer CSVs or a ZIP exported by the Rayleigh notebook.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if min(args.num_tx, args.train_samples, args.epochs, args.batch_size, args.hidden_layers) < 1:
        parser.error("Transmit streams, sample count, epochs, batch size, and depth must be positive.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive.")
    if not 0 <= args.seed < 2**64 - 1:
        parser.error("Seed must be in the range [0, 2**64 - 2].")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Select a GPU runtime in Colab or use --device cpu.")
    if args.output_dir == DEFAULT_OUTPUT_DIR and not Path("/content/drive/MyDrive").is_dir():
        parser.error("Mount Google Drive at /content/drive or pass --output-dir.")
    return args


def main(argv=None) -> Path:
    args = parse_args(argv)
    data_path = resolve_layer_data(args.layer_data, args.num_tx)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    constellation, bit_labels = constellation_for(args.modulation)
    model_kwargs = {
        "num_tx": args.num_tx,
        "modulation_order": len(constellation),
        "hidden_layers": args.hidden_layers,
    }
    model = PMAPNet(**model_kwargs).to(device)
    print(
        f"Loading {args.train_samples:,} {args.modulation.upper()} training vectors from {data_path}.",
        flush=True,
    )
    features, targets = load_layer_data(
        data_path, args.num_tx, constellation, args.train_samples, args.seed, device
    )

    run_dir = args.output_dir / datetime.now(timezone.utc).strftime("pmapnet_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    print(f"Using {device}. Saving to {run_dir}", flush=True)

    history = fit(model, features, targets, args, generator)
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "model_kwargs": model_kwargs,
        "constellation": constellation,
        "bit_labels": bit_labels,
        "settings": {
            "train_samples": args.train_samples,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "layer_data": str(data_path),
        },
        "training_mse_by_epoch": history,
    }
    torch.save(checkpoint, run_dir / f"pmapnet_{args.modulation}.pt")
    save_loss_plot(history, args.modulation, run_dir / "layer_loss.png")
    print(f"Saved checkpoint and loss figure in {run_dir}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()
