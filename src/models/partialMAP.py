import csv
from pathlib import Path

import torch


class partialMAP:
    def __init__(self, *, verbose=True, csv_log_dir=None):
        self.verbose = verbose
        self.csv_log_dir = Path(csv_log_dir) if csv_log_dir is not None else None
        self.csi = None
        self.possibleTXSymbols = None
        self.noiseVar = None
        self.Q = self.R = None
        self.bits = None
        self.M = self.B = None

    def updateCSI(self, csi):
        self.csi = torch.as_tensor(csi, dtype=torch.complex128)
        self.NR, self.NT = self.csi.shape
        self.Q, self.R = torch.linalg.qr(self.csi, mode="reduced")
        if self.verbose:
            print("csi updated:")
            print("H = ", self.csi)
            print("Q = ", self.Q)
            print("R = ", self.R)

    def updateNoiseVar(self, noiseVar):
        self.noiseVar = float(noiseVar)

    def updatePossibleTXSymbols(self, symbols, bits):
        labels = torch.as_tensor(bits)
        symbols = torch.as_tensor(symbols)
        self.possibleTXSymbols = torch.as_tensor(symbols, dtype=torch.complex128)
        self.bits = labels.long()
        self.M, self.B = labels.shape

    def _local_LLR(self, z, v, layer):
        k = self.NT - layer
        alphabet = torch.arange(self.M, device=v.device)
        grid = torch.cartesian_prod(*([alphabet] * k)).reshape(-1, k)
        projected = v @ self.possibleTXSymbols.to(v.device)[grid].T
        candidate_bits = self.bits.to(v.device)[grid].flatten(1)
        partitions = [
            (torch.where(column != 1)[0], torch.where(column == 1)[0])
            for column in candidate_bits.unbind(dim=1)
        ]

        # Keep the sample-by-candidate score tensor bounded.
        batch_size = max(1, 1_048_576 // projected.numel())
        sigma_local = z.real.new_empty((z.numel(), k * self.B))
        for samples, local in zip(z.split(batch_size), sigma_local.split(batch_size)):
            scores = -(samples[:, None] - projected).abs().square() / self.noiseVar
            for bit, (zeros, ones) in enumerate(partitions):
                log_p0 = scores.index_select(1, zeros).logsumexp(dim=1)
                log_p1 = scores.index_select(1, ones).logsumexp(dim=1)
                local[:, bit] = log_p0 - log_p1

        sigma_local = sigma_local.reshape(z.numel(), k, self.B)
        if self.verbose:
            print("Sigma local: ", sigma_local)
        return sigma_local

    def _log_layer_csv(self, layer, z, r, local_llr):
        """Append this layer's inputs and local bit LLRs, one row per sample."""
        path = self.csv_log_dir / f"layer_{layer}.csv"
        header = [f"z{layer}_re", f"z{layer}_im"]
        for tx in range(layer, self.NT):
            header.extend([f"r{layer}{tx}_re", f"r{layer}{tx}_im"])
        header.extend(
            f"llr_tx{tx}_bit{bit}"
            for tx in range(layer, self.NT)
            for bit in range(self.B)
        )

        r_fields = [
            part
            for value in r.detach().cpu().tolist()
            for part in (value.real, value.imag)
        ]
        z_values = z.detach().cpu().tolist()
        llr_values = local_llr.detach().flatten(1).cpu().tolist()

        with path.open("a+", encoding="utf-8", newline="") as csv_file:
            csv_file.seek(0)
            existing_header = next(csv.reader(csv_file), None)
            if existing_header is not None and existing_header != header:
                raise ValueError(
                    f"CSV columns do not match the detector configuration: {path}"
                )
            csv_file.seek(0, 2)
            writer = csv.writer(csv_file)
            if existing_header is None:
                writer.writerow(header)
            writer.writerows(
                [sample.real, sample.imag, *r_fields, *bit_llrs]
                for sample, bit_llrs in zip(z_values, llr_values)
            )

    def detectSequence(self, recieved):
        received = torch.as_tensor(recieved, dtype=self.csi.dtype, device=self.csi.device)
        z = received @ self.Q.conj()
        llr = z.real.new_zeros((z.shape[0], self.NT, self.B))

        if self.csv_log_dir is not None:
            self.csv_log_dir.mkdir(parents=True, exist_ok=True)

        for layer in range(self.NT - 1, -1, -1):
            z_layer = z[:, layer]
            r_layer = self.R[layer, layer:]
            local_llr = self._local_LLR(z_layer, r_layer, layer)
            llr[:, layer:] += local_llr
            if self.csv_log_dir is not None:
                self._log_layer_csv(layer, z_layer, r_layer, local_llr)
            if self.verbose:
                print(f"currect layer: {layer}")
                print(f"decoded LLR (Lamda):")
                print(llr)
                print("\n")

        return (llr < 0).long().transpose(0, 1)
