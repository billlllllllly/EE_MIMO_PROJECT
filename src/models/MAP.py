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
        prior = torch.as_tensor(prior, dtype=torch.float64)
        if (prior.ndim != 1 or not torch.isfinite(prior).all()
                or torch.any(prior < 0) or not torch.any(prior > 0)):
            raise ValueError("Prior must be a nonnegative finite vector with positive mass.")
        self.prior = prior

    def updateCSI(self, csi):
        csi = torch.as_tensor(csi, dtype=torch.complex128)
        if csi.ndim != 2 or 0 in csi.shape or not torch.isfinite(csi).all():
            raise ValueError("CSI must be a finite nonempty (N_rx, N_tx) matrix.")
        self.csi = csi
        if self.possibleTXSymbols is not None:
            self.possibleTXSymbols = self.possibleTXSymbols.to(csi.device)

    def updatePossibleTXSymbols(self, s):
        s = torch.as_tensor(s, dtype=torch.complex128,
                            device=None if self.csi is None else self.csi.device)
        if s.ndim not in (1, 2) or s.numel() == 0 or not torch.isfinite(s).all():
            raise ValueError("Symbols must be a finite nonempty constellation or dictionary.")
        self.possibleTXSymbols = s

    def _updatePossibleRXSymbols(self):
        if (self.csi is None or self.possibleTXSymbols is None):
            return
        self.possibleRXSymbols = self.csi @ self.possibleTXSymbols

    def updateNoiseVar(self, noiseVar):
        noiseVar = float(noiseVar)
        if not math.isfinite(noiseVar) or noiseVar < 0:
            raise ValueError("Noise variance must be finite and nonnegative.")
        self.noiseVar = noiseVar

    def detectSequence(self, received):
        """Input (N_rx,) or (N_samples, N_rx); output (N_tx,) or (N_tx, N_samples)."""
        self._validate_configuration()
        received = torch.as_tensor(received, dtype=self.csi.dtype, device=self.csi.device)
        if received.ndim not in (1, 2) or received.shape[-1] != self.csi.shape[0]:
            raise ValueError("Received input must have shape (N_rx,) or (N_samples, N_rx).")
        if not torch.isfinite(received).all():
            raise ValueError("Received input must contain finite values.")
        if received.ndim == 1:
            return self._detect_one(received)
        if received.shape[0] == 0:
            return received.new_empty((self.csi.shape[1], 0))
        return torch.stack([self._detect_one(y) for y in received], dim=1)

    def _validate_configuration(self):
        if self.csi is None:
            raise ValueError("CSI is None")
        if self.noiseVar is None:
            raise ValueError("Noise is None")
        if self.possibleRXSymbols is None:
            self.updatePossibleRXSymbols()
        if self.prior is not None and self.prior.shape != (self.possibleTXSymbols.shape[1],):
            raise ValueError("Prior must have one entry per dictionary column.")