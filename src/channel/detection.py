"""Monte Carlo BER testbench for batched flat-fading MIMO channels."""

import math
import operator

import torch


class Testbench:
    """Configure a BER sweep once and run it on any compatible CSI batch.

    ``models`` maps detector names to instances implementing ``updateCSI``,
    ``updatePossibleTXSymbols``, ``updateNoiseVar``, and ``detectSequence``.
    ``symbols`` is the shared constellation of shape (M,), and ``bit_labels``
    contains its corresponding binary labels of shape (M, bits_per_symbol).
    ``symbols_per_channel`` counts MIMO symbol vectors per CSI matrix. Each
    vector contains one independently sampled constellation symbol per stream.
    ``snr_db`` is a sequence of SNR points referenced to unit transmit-symbol
    energy, so use a unit-average-energy constellation for an Es/N0 sweep.
    """

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
    def run(self, csi, *, generator=None):
        """Return (ber, bits_per_snr) for CSI of shape (batch, nr, nt).

        ``ber`` is a tensor of shape (len(snr_db), len(models)), in the input
        order of the SNR points and detector names. Pass a torch.Generator on
        the CSI device to reproduce the symbol and AWGN draws.
        """
        csi = torch.as_tensor(csi, dtype=torch.complex128)
        if csi.ndim != 3 or any(size == 0 for size in csi.shape):
            raise ValueError("csi must have shape (batch, nr, nt) with positive dimensions.")
        batch, nr, nt = csi.shape
        symbols = self.symbols.to(csi.device)
        labels = self.bit_labels.to(csi.device)
        detectors = tuple(self.models.items())
        errors = torch.zeros(
            (len(self.snr_db), len(detectors)), dtype=torch.long, device=csi.device
        )

        for _, detector in detectors:
            detector.updatePossibleTXSymbols(symbols, labels)

        for i in csi.shape[0]-1:
            channel = csi[i]
            print(f"current running csi[{i}]")
            tx_indices = torch.randint(
                symbols.numel(), (nt, self.symbols_per_channel),
                device=csi.device, generator=generator,
            )
            tx_bits = labels[tx_indices]
            clean = channel @ symbols[tx_indices]
            noise = torch.randn(
                (nr, self.symbols_per_channel), dtype=csi.dtype,
                device=csi.device, generator=generator,
            )

            for _, detector in detectors:
                detector.updateCSI(channel)

            for i, variance in enumerate(self.noise_variances):
                # Complex randn has unit average power. Share samples across detectors and SNRs.
                received = (clean + math.sqrt(variance) * noise).T
                for j, (name, detector) in enumerate(detectors):
                    detector.updateNoiseVar(variance)
                    detected_bits = detector.detectSequence(received)
                    if detected_bits.shape != tx_bits.shape:
                        raise ValueError(
                            f"{name} returned bits with shape {tuple(detected_bits.shape)}. "
                            f"Expected {tuple(tx_bits.shape)} (nt, symbols_per_channel, bits_per_symbol)."
                        )
                    errors[i, j] += (detected_bits != tx_bits).count_nonzero()

        bits_per_snr = batch * nt * self.symbols_per_channel * labels.shape[1]
        return errors.double() / bits_per_snr, bits_per_snr
