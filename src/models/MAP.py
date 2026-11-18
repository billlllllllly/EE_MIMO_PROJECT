import math
import torch

class MAP:
    def __init__(self):
        self.csi = None
        self.possibleTXSymbols = None
        self.noiseVar = None
        self.possibleRXSymbols = None
        self.prior = None

    def updatePrior(self, prior):
        self.prior = torch.as_tensor(prior, dtype=torch.float64)

    def updateCSI(self, csi):
        self.csi = torch.as_tensor(csi, dtype=torch.complex128)

    def updatePossibleTXSymbols(self, s):
        self.possibleTXSymbols = torch.as_tensor(s, dtype=torch.complex128)

    def _updatePossibleRXSymbols(self):
        self.possibleRXSymbols = self.csi @ self.possibleTXSymbols

    def updateNoiseVar(self, noiseVar):
        self.noiseVar = float(noiseVar)

    def _detect_one(self, y):
        squared_dist = (y[:, None] - self.possibleRXSymbols).abs().square().sum(dim=0)
        log_prior = 0.0
        if self.prior is not None:
            log_prior = self.prior.to(y.device).log()
        index = (-squared_dist / self.noiseVar + log_prior).argmax()
        return self.possibleTXSymbols[:, index]

    def detectSequence(self, received):
        received = torch.as_tensor(received, dtype=self.csi.dtype, device=self.csi.device)
        return torch.stack([self._detect_one(y) for y in received], dim=1)