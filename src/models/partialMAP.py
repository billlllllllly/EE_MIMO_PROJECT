import math
import torch

class partialMAP:
    def __init__(self):
        self.csi = None
        self.possibleTXSymbols = None
        self.noiseVar = None
        self.Q, self.R = None
        self.bits = None
        self.M, self.B = None

    def _configuration_changed(self, parameter):
        if parameter == "csi":
            self.NR, self.NT = self.csi.shape
            self.Q, self.R = torch.linalg.qr(self.csi, mode="reduced")

    def updatePossibleTXSymbols(self, symbols, bits):
        labels = torch.as_tensor(bits)
        symbols = torch.as_tensor(symbols)
        if (symbols.ndim != 1 or labels.ndim != 2 or labels.shape[0] != symbols.numel()
                or labels.shape[1] == 0 or symbols.numel() != 2 ** labels.shape[1]
                or not ((labels == 0) | (labels == 1)).all()
                or torch.unique(labels, dim=0).shape[0] != symbols.numel()):
            raise ValueError("Symbol label error.")
        s = torch.as_tensor(symbols, dtype=torch.complex128,
                            device=None if self.csi is None else self.csi.device)
        if s.ndim not in (1, 2) or s.numel() == 0 or not torch.isfinite(s).all():
            raise ValueError("Symbols must be a finite nonempty constellation or dictionary.")
        self.possibleTXSymbols = s
        self.bits = labels.long()
        self.M, self.B = labels.shape

    def _validate_configuration(self):
        if self.csi is None:
            raise ValueError("CSI must be updated before detection.")
        s = self.possibleTXSymbols
        if s is not None and s.ndim == 2 and s.shape[0] != self.csi.shape[1]:
            raise ValueError("Dictionary must have one row per transmit stream.")
        if self.NR < self.NT:
            raise ValueError("Partial MAP requires N_rx >= N_tx.")
        if self.bits is None or self.noiseVar is None or self.noiseVar <= 0:
            raise ValueError("Partial MAP needs a labeled constellation and positive noise variance.")

    def _local_LLR(self, z, v, layer):
        k = self.NT - layer
        alphabet = torch.arange(self.M, device=v.device)
        grid = torch.cartesian_prod(*([alphabet] * k)).reshape(-1, k)
        # constraint on reshape, it must be (-1, k)

        S_all = self.possibleTXSymbols.to(v.device)[grid]  # (M**k, k)
        B_all = self.bits.to(v.device)[grid]      # (M**k, k, B)

        # One log likelihood per candidate vector, with uniform priors.
        scores = -torch.abs(z - v @ S_all.T).square() / self.noiseVar

        Sigma_local = torch.empty((k, self.B), dtype=scores.dtype, device=v.device)
        for j in range(k):
            for b in range(self.B):
                mask1 = (B_all[:, j, b] == 1)
                log_p0 = torch.logsumexp(scores[~mask1], dim=0)
                log_p1 = torch.logsumexp(scores[mask1], dim=0)
                Sigma_local[j, b] = log_p0 - log_p1
        return Sigma_local

    def _detect_one(self, y):
        z = self.Q.mH @ y
        Theta = None

        for layer in range(self.NT - 1, -1, -1):
            Sigma_local = self._local_LLR(z[layer], self.R[layer, layer:], layer)
            if Theta is not None:
                # The previous layer covers only streams layer+1 through N_tx-1.
                Sigma_local[1:] += Theta
            Theta = Sigma_local

        hard_bits = (Theta < 0).long()
        labels = self.bits.to(y.device)
        matches = (hard_bits[:, None, :] == labels[None, :, :]).all(dim=-1)
        indices = matches.long().argmax(dim=1)
        return self.possibleTXSymbols.to(y.device)[indices]
        

    def detect_sequesnce(self, received):
        """Compatibility with the original method spelling."""
        return self.detect_sequence(received)

    def updateCSI(self, csi):
        csi = torch.as_tensor(csi, dtype=torch.complex128)
        if csi.ndim != 2 or 0 in csi.shape or not torch.isfinite(csi).all():
            raise ValueError("CSI must be a finite nonempty (N_rx, N_tx) matrix.")
        self.csi = csi
        if self.possibleTXSymbols is not None:
            self.possibleTXSymbols = self.possibleTXSymbols.to(csi.device)
        self._configuration_changed("csi")

    def updateNoiseVar(self, noiseVar):
        noiseVar = float(noiseVar)
        if not math.isfinite(noiseVar) or noiseVar < 0:
            raise ValueError("Noise variance must be finite and nonnegative.")
        self.noiseVar = noiseVar

    def detectSequence(self, recieved):
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
