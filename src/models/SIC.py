import torch

class SIC:
    """Unsorted ZF-QR-SIC: detect streams N_tx-1, ..., 1, 0.
    Requires N_rx >= N_tx and a full-column-rank channel. QR is cached per CSI.
    """

    def __init__(self):
        self.CSI = None
        self.possibleTXSymbols = None
        self.bits = None
        self.noiseVar = None
        self.Q = None
        self.R = None

    def updateCSI(self, csi):
        self.CSI = torch.as_tensor(csi, dtype=torch.complex128)
        self.NR, self.NT = self.CSI.shape
        self.Q, self.R = torch.linalg.qr(self.CSI, mode="reduced")

    def updatePossibleTXSymbols(self, s, bits):
        self.possibleTXSymbols = torch.as_tensor(s, dtype=torch.complex128)
        self.bits = torch.as_tensor(bits, dtype=torch.long)

    def updateNoiseVar(self, noiseVar):
        noiseVar = float(noiseVar)
        self.noiseVar = noiseVar

    def _detect_one(self, y):
        z = self.Q.mH @ y
        # .mH for matrix Hermitian
        symbols = self.possibleTXSymbols.to(y.device)
        x_hat = y.new_zeros(self.NT)
        indices = torch.empty(self.NT, dtype=torch.long, device=y.device)

        for k in range(self.NT-1, -1, -1):
            # z[k] = [0, 0, ..., rkk, rkk+1, ... rkNT] * [x, x, xkk, ...]
            interference = self.R[k, k + 1:] @ x_hat[k + 1:]
            estimate = (z[k] - interference) / self.R[k, k]
            # hard output
            dist = torch.abs(estimate.unsqueeze(-1) - symbols)
            indices[k] = dist.argmin()
            x_hat[k] = symbols[indices[k]]

        return self.bits.to(y.device)[indices]

    def detectSequence(self, received):
        """Return (N_tx, B) or (N_tx, N_samples, B) bits."""
        received = torch.as_tensor(received, dtype=self.CSI.dtype, device=self.CSI.device)
        if received.ndim == 1:
            return self._detect_one(received)
        return torch.stack([self._detect_one(y) for y in received], dim=1)
