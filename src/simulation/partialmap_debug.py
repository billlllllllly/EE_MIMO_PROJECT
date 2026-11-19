import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.channel.rayleighfading import rayleighfading
from src.models.partialMAP import partialMAP


def main():
    torch.manual_seed(0)
    csi = rayleighfading(NT=2, NR=2).to(torch.complex128)
    qpsk = torch.tensor([-1 - 1j, -1 + 1j, 1 + 1j, 1 - 1j], dtype=csi.dtype)
    gray = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]])

    detector = partialMAP()
    detector.updateCSI(csi=csi)
    detector.updatePossibleTXSymbols(symbols=qpsk, bits=gray)
    detector.updateNoiseVar(noiseVar=0.01)

    tx_idx = torch.tensor([1, 2])
    tx_bits = gray[tx_idx]
    # One received sample with one value per receive antenna.
    y_clean = (csi @ qpsk[tx_idx]).unsqueeze(0)

    detected_bits = detector.detectSequence(y_clean)[:, 0, :]
    print("Transmitted bits:", tx_bits.tolist())
    print("Detected bits:   ", detected_bits.tolist())
    torch.testing.assert_close(detected_bits, tx_bits)


if __name__ == "__main__":
    main()
