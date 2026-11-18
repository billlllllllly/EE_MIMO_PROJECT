import torch

def rayleighfading(NR, NT):
    return torch.randn(NR, NT, dtype=torch.cfloat)