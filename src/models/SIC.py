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

    def detectSequence(self, received):
        """Return (N_tx, B) or (N_tx, N_samples, B) bits."""
        received = torch.as_tensor(received, dtype=self.CSI.dtype, device=self.CSI.device)
        z = self.Q.mH @ torch.atleast_2d(received).T
        symbols = self.possibleTXSymbols.to(received.device)
        x_hat = torch.zeros_like(z)
        indices = torch.empty_like(z, dtype=torch.long)

        # Each stream depends on the decisions for higher streams in the same sample.
        for k in range(self.NT - 1, -1, -1):
            interference = self.R[k, k + 1:] @ x_hat[k + 1:]
            estimate = (z[k] - interference) / self.R[k, k]
            indices[k] = (estimate.unsqueeze(-1) - symbols).abs().argmin(dim=-1)
            x_hat[k] = symbols[indices[k]]

        detected = self.bits.to(received.device)[indices]
        return detected[:, 0] if received.ndim == 1 else detected
