import torch

class MMSE:
    def __init__(self):
        self.csi = None
        self.possibleTXSymbols = None
        self.bits = None
        self.noiseVar = None
        self.txVar = 1.0
        self.W = None

    def updateTxVar(self, txVar):
        self.txVar = float(txVar)
        self._compute_weights()

    def _compute_weights(self):
        self.W = None
        if self.csi is not None and self.noiseVar is not None:
            h = self.csi
            gram = h.conj().T @ h + (self.noiseVar / self.txVar) * torch.eye(h.shape[1], dtype=h.dtype, device=h.device)
            self.W = torch.linalg.solve(gram, h.conj().T)

    def _detect_one(self, y):
        estimate = self.W @ y
        gain = torch.diag(self.W @ self.csi)
        estimate = estimate / torch.where(gain != 0, gain, 1)
        estimate = torch.where(gain != 0, estimate, 0)
        return self._slice_vector(estimate)

    def updateCSI(self, csi):
        self.csi = torch.as_tensor(csi, dtype=torch.complex128)
        self._compute_weights()

    def updatePossibleTXSymbols(self, s, bits):
        self.possibleTXSymbols = torch.as_tensor(s, dtype=torch.complex128)
        self.bits = torch.as_tensor(bits, dtype=torch.long)

    def updateNoiseVar(self, noiseVar):
        self.noiseVar = float(noiseVar)
        self._compute_weights()

    def _slice_vector(self, estimate):
        s = self.possibleTXSymbols.to(estimate.device)
        bits = self.bits.to(estimate.device)
        distance = (estimate[:, None] - s).abs().square()
        if s.ndim == 1:
            return bits[distance.argmin(dim=1)]
        return bits[:, distance.sum(dim=0).argmin()]

    def detect_sequence(self, received):
        """Return (N_tx, B) or (N_tx, N_samples, B) bits."""
        received = torch.as_tensor(received, dtype=self.csi.dtype, device=self.csi.device)
        if received.ndim == 1:
            return self._detect_one(received)
        if received.shape[0] == 0:
            return self.bits.new_empty(
                (self.csi.shape[1], 0, self.bits.shape[-1]), device=received.device
            )
        return torch.stack([self._detect_one(y) for y in received], dim=1)

    def detectSequence(self, recieved):
        """Compatibility with existing scripts and the original keyword spelling."""
        return self.detect_sequence(recieved)
