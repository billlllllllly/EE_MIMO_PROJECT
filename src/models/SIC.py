import math
import torch

class SIC:
    """Unsorted ZF-QR-SIC: detect streams N_tx-1, ..., 1, 0.
    Requires N_rx >= N_tx and a full-column-rank channel. QR is cached per CSI.
    """

    def __init__(self):
        self.CSI = None
        self.possibleTXSymbols = None
        self.noiseVar = None
        self.Q = None
        self.R = None

    def updateCSI(self, csi):
        self.CSI = torch.as_tensor(csi, dtype=torch.complex128)
        self.NR, self.NT = self.CSI.shape
        self.Q, self.R = torch.linalg.qr(self.CSI, mode="reduced")

    def updatePossibleTXSymbols(self, s):
        self.possibleTXSymbols = torch.as_tensor(s, dtype=torch.complex128)

    def updateNoiseVar(self, noiseVar):
        noiseVar = float(noiseVar)
        self.noiseVar = noiseVar

    def _detect_one(self, y):
        z = self.Q.mH @ y
        # .mH for matrix Hermitian
        x_hat = torch.zeros(self.NT, dtype=torch.cfloat)

        for k in range(self.NT-1, -1, -1):
            # z[k] = [0, 0, ..., rkk, rkk+1, ... rkNT] * [x, x, xkk, ...]
            interference = self.R[k, k + 1:] @ x_hat[k + 1:]
            estimate = (z[k] - interference) / self.R[k, k]
            # hard output
            dist = torch.abs(estimate.unsqueeze(-1) - self.possibleTXSymbols)
            x_hat[k] = self.possibleTXSymbols[torch.argmin(dist, dim=-1)]

        return x_hat

    def detectSequence(self, received):
        """Input (N_rx,) or (N_samples, N_rx); output (N_tx,) or (N_tx, N_samples)."""
        self._validate_configuration()
        received = torch.as_tensor(received, dtype=self.CSI.dtype, device=self.CSI.device)
        if received.ndim not in (1, 2) or received.shape[-1] != self.CSI.shape[0]:
            raise ValueError("Received input must have shape (N_rx,) or (N_samples, N_rx).")
        if not torch.isfinite(received).all():
            raise ValueError("Received input must contain finite values.")
        if received.ndim == 1:
            return self._detect_one(received)
        if received.shape[0] == 0:
            return received.new_empty((self.CSI.shape[1], 0))
        return torch.stack([self._detect_one(y) for y in received], dim=1)
