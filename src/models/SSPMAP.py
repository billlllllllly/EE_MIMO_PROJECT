"""Soft-SIC aided partial MAP detector (SS-PMAP, Variant A)."""

import math

import torch

from .SIC import SortedQRSIC


class SSPMAP:
    """Sorted soft-SIC detection with approximate pairwise LLR refinement.

    Set the shared constellation, CSI and complex noise variance using the
    testbench update methods. LLRs use log(P(bit=0) / P(bit=1)).
    """

    def __init__(self, *, alpha=0.85, llr_max=20.0):
        self.alpha = float(alpha)
        self.llr_max = float(llr_max)
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        if not math.isfinite(self.llr_max) or self.llr_max <= 0:
            raise ValueError("llr_max must be finite and positive.")
        self.Q = self.R = self.order = None
        self._inverse_order = None
        self.possibleTXSymbols = None
        self.noiseVar = None

    def updateCSI(self, csi):
        channel = torch.as_tensor(csi, dtype=torch.complex128)
        if channel.ndim != 2 or 0 in channel.shape or not torch.isfinite(channel).all():
            raise ValueError("CSI must be a finite, nonempty (N_rx, N_tx) matrix.")
        if channel.shape[0] < channel.shape[1]:
            raise ValueError("SS-PMAP requires N_rx >= N_tx.")
        # Householder QR keeps Q orthonormal even when a pivot is zero.
        # The likelihoods use R directly and never divide by a diagonal entry.
        self.Q, self.R, self.order = SortedQRSIC._sorted_qr(channel)
        self._inverse_order = self.order.argsort()

    def updateNoiseVar(self, noiseVar):
        variance = float(noiseVar)
        if not math.isfinite(variance) or variance <= 0:
            raise ValueError("Complex noise variance must be finite and positive.")
        self.noiseVar = variance

    def updatePossibleTXSymbols(self, symbols, bits):
        symbols = torch.as_tensor(symbols, dtype=torch.complex128)
        labels = torch.as_tensor(bits, device=symbols.device)
        if symbols.ndim != 1 or symbols.numel() == 0 or not torch.isfinite(symbols).all():
            raise ValueError("symbols must be a finite, nonempty vector of shape (M,).")
        if labels.ndim != 2 or labels.shape[0] != symbols.numel() or labels.shape[1] == 0:
            raise ValueError("bits must have shape (M, bits_per_symbol).")
        if labels.is_complex() or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("bits must contain binary labels.")
        self.possibleTXSymbols = symbols
        self._signs = 1.0 - 2.0 * labels.to(torch.float64)
        self._partitions = [
            ((col == 0).nonzero().flatten(), (col == 1).nonzero().flatten())
            for col in labels.T
        ]

    @staticmethod
    def _bit_llrs(scores, partitions):
        """Marginalize symbol log weights over each supplied bit partition."""
        return torch.stack([
            scores.index_select(-1, zeros).logsumexp(-1)
            - scores.index_select(-1, ones).logsumexp(-1)
            for zeros, ones in partitions
        ], dim=-1)

    def _detect_batch(self, z, symbols, signs, partitions):
        num_samples, nt = z.shape
        llrs = z.real.new_empty((num_samples, nt, len(partitions)))
        noise = max(self.noiseVar, 1e-12)
        bottom = -(z[:, -1, None] - self.R[-1, -1] * symbols).abs().square() / noise
        llrs[:, -1] = self._bit_llrs(bottom, partitions).clamp(-self.llr_max, self.llr_max)

        for k in range(nt - 2, -1, -1):
            # All priors and moments for this row come from the previous row.
            prior = (0.5 * llrs[:, k + 1:]) @ signs.T
            prob = prior.softmax(dim=-1)
            mean = (prob * symbols).sum(dim=-1)
            variance = (prob * (symbols - mean[..., None]).abs().square()).sum(dim=-1)
            gains = self.R[k, k + 1:]
            interference = gains * mean
            power = gains.abs().square() * variance
            total_mean = interference.sum(dim=-1)
            total_power = power.sum(dim=-1)
            row = z[:, k]
            desired = self.R[k, k] * symbols

            # Variant A soft-cancels every previously detected stream in Step 2B.
            residual = row - total_mean
            effective_noise = (self.noiseVar + total_power).clamp_min(1e-12)
            scores = -(residual[:, None] - desired).abs().square() / effective_noise[:, None]
            llrs[:, k] = self._bit_llrs(scores, partitions).clamp(-self.llr_max, self.llr_max)

            for j, gain in enumerate(gains):
                # Keep this interferer in the pair and soft-cancel the others.
                residual = row - (total_mean - interference[:, j])
                pair_noise = (self.noiseVar + (total_power - power[:, j])).clamp_min(1e-12)
                pair_symbols = desired[:, None] + gain * symbols[None, :]
                distances = (residual[:, None, None] - pair_symbols).abs().square()
                message = (-distances / pair_noise[:, None, None]).logsumexp(dim=1)
                pair_llr = self._bit_llrs(message + prior[:, j], partitions)
                # pair_llr includes the old target-bit prior, so its difference
                # from old is the extrinsic increment in Eqs. 23-25.
                old = llrs[:, k + 1 + j]
                llrs[:, k + 1 + j] = (
                    (1.0 - self.alpha) * old + self.alpha * pair_llr
                ).clamp(-self.llr_max, self.llr_max)
        return llrs

    @torch.no_grad()
    def detect_llrs(self, received):
        """Return bit LLRs in transmit order, (N_tx, samples, B) or (N_tx, B)."""
        if self.Q is None or self.possibleTXSymbols is None or self.noiseVar is None:
            raise ValueError("Set CSI, constellation with bit labels, and noise variance before detection.")
        received = torch.as_tensor(received, dtype=self.Q.dtype, device=self.Q.device)
        nr, nt = self.Q.shape
        if received.ndim not in (1, 2) or received.shape[-1] != nr:
            raise ValueError("received must have shape (N_rx,) or (samples, N_rx).")
        symbols = self.possibleTXSymbols.to(received.device)
        signs = self._signs.to(received.device)
        partitions = [(a.to(received.device), b.to(received.device)) for a, b in self._partitions]
        z = torch.atleast_2d(received) @ self.Q.conj()
        output = z.real.new_empty((nt, z.shape[0], len(partitions)))
        # Bound the sample-by-M-by-M distance tensor and the soft moment tables.
        m = symbols.numel()
        terms = max(m * m if nt > 1 else m, (nt - 1) * m)
        batch_size = max(1, 1_048_576 // terms)
        for start in range(0, z.shape[0], batch_size):
            batch = self._detect_batch(z[start:start + batch_size], symbols, signs, partitions)
            output[:, start:start + batch_size] = batch.transpose(0, 1).index_select(0, self._inverse_order)
        return output[:, 0] if received.ndim == 1 else output

    def detectSequence(self, received):
        """Return integer hard bits in the same shape as detect_llrs."""
        return (self.detect_llrs(received) < 0).long()
