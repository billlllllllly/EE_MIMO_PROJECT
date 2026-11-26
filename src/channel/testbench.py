"""Monte Carlo BER testbench for batched flat-fading MIMO channels."""

import csv
import math
import operator

import torch


class Testbench:
    def __init__(self, models, symbols, bit_labels, symbols_per_channel, snr_db):
        self.models = dict(models)
        if not self.models:
            raise ValueError("At least one detector is required.")

        self.symbols = torch.as_tensor(symbols, dtype=torch.complex128)
        if self.symbols.ndim != 1 or self.symbols.numel() == 0:
            raise ValueError("symbols must be a nonempty shared constellation of shape (M,).")

        labels = torch.as_tensor(bit_labels)
        if labels.ndim != 2 or labels.shape[0] != self.symbols.numel() or labels.shape[1] == 0:
            raise ValueError("bit_labels must have shape (M, bits_per_symbol).")
        if labels.is_complex() or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("bit_labels must contain binary values.")
        self.bit_labels = labels.long()

        self.symbols_per_channel = operator.index(symbols_per_channel)
        if self.symbols_per_channel <= 0:
            raise ValueError("symbols_per_channel must be positive.")

        self.snr_db = tuple(float(snr) for snr in snr_db)
        if not self.snr_db or not all(math.isfinite(snr) for snr in self.snr_db):
            raise ValueError("snr_db must contain at least one SNR and all SNRs must be finite.")
        try:
            self.noise_variances = tuple(10.0 ** (-snr / 10.0) for snr in self.snr_db)
        except OverflowError as exc:
            raise ValueError("snr_db must yield finite, positive noise variances.") from exc
        if any(variance == 0 for variance in self.noise_variances):
            raise ValueError("snr_db must yield finite, positive noise variances.")

    @torch.no_grad()
    def run(self, csi, *, generator=None, csv_path=None):
        """Return BER and bits per SNR, optionally writing counts by transmit antenna."""
        csi = torch.as_tensor(csi, dtype=torch.complex128)
        if csi.ndim != 3 or any(size == 0 for size in csi.shape):
            raise ValueError("csi must have shape (batch, NR, NT) with positive dimensions.")
        batch, _, NT = csi.shape
        symbols = self.symbols.to(csi.device)
        # symbols have shape (M,)
        labels = self.bit_labels.to(csi.device)
        # labels have shape (M, B)
        detectors = tuple(self.models.items())
        errors = torch.zeros(
            (len(self.snr_db), len(detectors), NT), dtype=torch.long, device=csi.device
        )

        for _, detector in detectors:
            detector.updatePossibleTXSymbols(symbols, labels)

        progress_interval = max(1, batch // 100)
        for channel_index, channel in enumerate(csi):
            if channel_index % progress_interval == 0:
                print(f"running channel number {channel_index}")

            tx_indices = torch.randint(
                symbols.numel(), (NT, self.symbols_per_channel),
                device=csi.device, generator=generator,
            )
            # tx_indices have shape (NT, symbols_per_channel)
            tx_bits = labels[tx_indices]
            # tx_bits have shape (NT, symbols_per_channel, B)
            tx_symbols = symbols[tx_indices]
            # tx_symbols have shape (NT, symbols_per_channel)
            clean_rx_symbols = channel @ tx_symbols
            # clean_rx_symbols have shape (NR, symbols_per_channel)
            
            noise = torch.randn(
                clean_rx_symbols.shape, dtype=csi.dtype,
                device=csi.device, generator=generator,
            )
            # base noise with unit variance and same shape as rx_symbols

            for _, detector in detectors:
                detector.updateCSI(channel)

            for i, variance in enumerate(self.noise_variances):
                # Complex randn has unit average power. Share samples across detectors and SNRs.
                received = (clean_rx_symbols + math.sqrt(variance) * noise).T
                for j, (name, detector) in enumerate(detectors):
                    detector.updateNoiseVar(variance)
                    detected_bits = detector.detectSequence(received)
                    if detected_bits.shape != tx_bits.shape:
                        raise ValueError(
                            f"{name} returned bits with shape {tuple(detected_bits.shape)}. "
                            f"Expected {tuple(tx_bits.shape)} (nt, symbols_per_channel, bits_per_symbol)."
                        )
                    errors[i, j] += (detected_bits != tx_bits).count_nonzero(dim=(1, 2))

        bits_per_snr = batch * NT * self.symbols_per_channel * labels.shape[1]
        if csv_path is not None:
            self._write_error_csv(csv_path, errors, bits_per_snr)
        return errors.sum(dim=-1).double() / bits_per_snr, bits_per_snr

    def _write_error_csv(self, csv_path, errors, bits_per_snr):
        antenna_columns = [f"tx{index}_bit_errors" for index in range(1, errors.shape[-1] + 1)]
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["snr_db", "detector", *antenna_columns, "total_bits_transmitted"])
            for snr, detector_counts in zip(self.snr_db, errors.tolist()):
                for name, antenna_counts in zip(self.models, detector_counts):
                    writer.writerow([snr, name, *antenna_counts, bits_per_snr])
