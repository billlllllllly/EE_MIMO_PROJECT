import math
import torch
# my codex

class MMSE:
    def __init__(self):
        self.csi = None
        self.possibleTXSymbols = None
        self.noiseVar = None
        self.txVar = 1.0
        self.W = None

    def _configuration_changed(self, parameter):
        if parameter in ("csi", "noise"):
            self._compute_weights()

    def updateTxVar(self, txVar):
        txVar = float(txVar)
        if not math.isfinite(txVar) or txVar <= 0:
            raise ValueError("Transmit variance must be finite and positive.")
        self.txVar = txVar
        self._compute_weights()

    def _compute_weights(self):
        self.W = None
        if self.csi is not None and self.noiseVar is not None:
            h = self.csi
            gram = h.conj().T @ h + (self.noiseVar / self.txVar) * torch.eye(h.shape[1], dtype=h.dtype, device=h.device)
            self.W = torch.linalg.solve(gram, h.conj().T)

    def _validate_configuration(self):
        if self.csi is None:
            raise ValueError("CSI must be updated before detection.")
        s = self.possibleTXSymbols
        if s is not None and s.ndim == 2 and s.shape[0] != self.csi.shape[1]:
            raise ValueError("Dictionary must have one row per transmit stream.")
        if self.W is None:
            raise ValueError("CSI and noise variance must be updated before MMSE detection.")

    def _detect_one(self, y):
        estimate = self.W @ y
        if self.possibleTXSymbols is None:
            return estimate
        gain = torch.diag(self.W @ self.csi)
        estimate = estimate / torch.where(gain != 0, gain, 1)
        estimate = torch.where(gain != 0, estimate, 0)
        return self._slice_vector(estimate)

    def updateCSI(self, csi):
        csi = torch.as_tensor(csi, dtype=torch.complex128)
        if csi.ndim != 2 or 0 in csi.shape or not torch.isfinite(csi).all():
            raise ValueError("CSI must be a finite nonempty (N_rx, N_tx) matrix.")
        self.csi = csi
        if self.possibleTXSymbols is not None:
            self.possibleTXSymbols = self.possibleTXSymbols.to(csi.device)
        self._configuration_changed("csi")

    def updatePossibleTXSymbols(self, s):
        s = torch.as_tensor(s, dtype=torch.complex128,
                            device=None if self.csi is None else self.csi.device)
        if s.ndim not in (1, 2) or s.numel() == 0 or not torch.isfinite(s).all():
            raise ValueError("Symbols must be a finite nonempty constellation or dictionary.")
        self.possibleTXSymbols = s
        self._configuration_changed("symbols")

    def updateNoiseVar(self, noiseVar):
        noiseVar = float(noiseVar)
        if not math.isfinite(noiseVar) or noiseVar < 0:
            raise ValueError("Noise variance must be finite and nonnegative.")
        self.noiseVar = noiseVar
        self._configuration_changed("noise")

    def _slice_vector(self, estimate):
        s = self.possibleTXSymbols
        if s is None:
            return estimate
        if s.ndim == 1:
            return s[(estimate[:, None] - s).abs().square().argmin(dim=1)]
        index = (estimate[:, None] - s).abs().square().sum(dim=0).argmin()
        return s[:, index]

    def detect_sequence(self, received):
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

    def detectSequence(self, recieved):
        """Compatibility with existing scripts and the original keyword spelling."""
        return self.detect_sequence(recieved)
