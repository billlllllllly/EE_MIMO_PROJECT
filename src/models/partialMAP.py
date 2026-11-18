import torch

class partialMAP:
    def __init__(self):
        self.csi = None
        self.possibleTXSymbols = None
        self.noiseVar = None
        self.Q = self.R = None
        self.bits = None
        self.M = self.B = None

    def updateCSI(self, csi):
        self.csi = torch.as_tensor(csi, dtype=torch.complex128)
        self.NR, self.NT = self.csi.shape
        self.Q, self.R = torch.linalg.qr(self.csi, mode="reduced")

    def updateNoiseVar(self, noiseVar):
        self.noiseVar = float(noiseVar)

    def updatePossibleTXSymbols(self, symbols, bits):
        labels = torch.as_tensor(bits)
        symbols = torch.as_tensor(symbols)
        self.possibleTXSymbols = torch.as_tensor(symbols, dtype=torch.complex128)
        self.bits = labels.long()
        self.M, self.B = labels.shape

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
        llr = z.real.new_zeros((self.NT, self.B))

        for layer in range(self.NT - 1, -1, -1):
            llr[layer:] += self._local_LLR(
                z[layer], self.R[layer, layer:], layer
            )

        return (llr < 0).long()

    def detectSequence(self, recieved):
        received = torch.as_tensor(recieved, dtype=self.csi.dtype, device=self.csi.device)
        return torch.stack([self._detect_one(y) for y in received], dim=1)
