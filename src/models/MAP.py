import torch

class MAP:
    def __init__(self):
        self.csi = None
        self.possibleTXSymbols = None
        self.noiseVar = None
        self.possibleRXSymbols = None
        self.prior = None
        self.bits = None
        self.candidateIndices = None

    def updatePrior(self, prior):
        self.prior = torch.as_tensor(prior, dtype=torch.float64)

    def updateCSI(self, csi):
        self.csi = torch.as_tensor(csi, dtype=torch.complex128)

    def updatePossibleTXSymbols(self, s, bits):
        self.possibleTXSymbols = torch.as_tensor(s, dtype=torch.complex128)
        self.bits = torch.as_tensor(bits, dtype=torch.long)

    def _updatePossibleRXSymbols(self):
        symbols = self.possibleTXSymbols.to(self.csi.device)
        num_streams = self.csi.shape[1]
        alphabet = torch.arange(symbols.numel(), device=self.csi.device)
        self.candidateIndices = torch.cartesian_prod(
            *([alphabet] * num_streams)
        ).reshape(-1, num_streams)
        self.possibleRXSymbols = self.csi @ symbols[self.candidateIndices].T

    def updateNoiseVar(self, noiseVar):
        self.noiseVar = float(noiseVar)

    def detectSequence(self, received):
        received = torch.as_tensor(received, dtype=self.csi.dtype, device=self.csi.device)
        self._updatePossibleRXSymbols()
        log_prior = 0.0 if self.prior is None else self.prior.to(received.device).log()

        # Bound the temporary sample-by-antenna-by-candidate tensor.
        batch_size = max(1, 1_048_576 // self.possibleRXSymbols.numel())
        indices = []
        for batch in received.split(batch_size):
            residual = batch.unsqueeze(-1) - self.possibleRXSymbols
            squared_dist = residual.abs().square().sum(dim=1)
            scores = -squared_dist / self.noiseVar + log_prior
            indices.append(scores.argmax(dim=-1))

        candidates = self.candidateIndices[torch.cat(indices)]
        return self.bits.to(received.device)[candidates].transpose(0, 1)
