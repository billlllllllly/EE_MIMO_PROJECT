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


class SortedQRSIC(SIC):
    """Sorted QR with the smallest residual-norm pivot, followed by ZF-SIC.

    The cached factors satisfy CSI[:, order] = Q @ R. Detection returns bits
    in the original transmit-stream order using the same interface as SIC.
    """

    def __init__(self):
        super().__init__()
        self.order = None
        self._inverse_order = None

    def updateCSI(self, csi):
        csi = torch.as_tensor(csi, dtype=torch.complex128)
        if csi.ndim != 2 or 0 in csi.shape or not torch.isfinite(csi).all():
            raise ValueError("CSI must be a finite nonempty (N_rx, N_tx) matrix.")
        n_rx, n_tx = csi.shape
        if n_rx < n_tx:
            raise ValueError("Sorted QR-SIC requires N_rx >= N_tx.")

        q, r, order = self._sorted_qr(csi)
        if torch.linalg.matrix_rank(r) != n_tx:
            raise ValueError("Sorted QR-SIC requires a full-column-rank channel.")

        self.CSI, self.Q, self.R = csi, q, r
        self.NR, self.NT = n_rx, n_tx
        self.order = order
        self._inverse_order = order.argsort()

    @staticmethod
    def _sorted_qr(csi):
        n_rx, n_tx = csi.shape
        r = csi.clone()
        order = torch.arange(n_tx, device=csi.device)
        reflectors = []

        for k in range(n_tx):
            # Previous reflectors compress the remaining residuals to rows k:.
            norms = torch.linalg.vector_norm(r[k:, k:], dim=0)
            pivot = k + norms.argmin().item()
            if pivot != k:
                columns = [k, pivot]
                r[:, columns] = r[:, columns[::-1]]
                order[columns] = order[columns[::-1]]

            norm = norms[pivot - k]
            if norm == 0:
                continue

            reflector = r[k:, k] / norm
            first = reflector[0]
            # Matching the leading entry's phase avoids cancellation.
            phase = torch.sgn(first) if first != 0 else 1
            reflector[0] += phase
            reflector = reflector / torch.linalg.vector_norm(reflector)
            trailing = r[k:, k:]
            trailing -= 2 * reflector[:, None] * (reflector.conj() @ trailing)
            reflectors.append((k, reflector))

        # Apply the Hermitian reflectors in reverse to form the reduced Q.
        q = torch.eye(n_rx, n_tx, dtype=csi.dtype, device=csi.device)
        for k, reflector in reversed(reflectors):
            trailing = q[k:, :]
            trailing -= 2 * reflector[:, None] * (reflector.conj() @ trailing)
        return q, torch.triu(r[:n_tx]), order

    def detectSequence(self, received):
        """Return (N_tx, B) or (N_tx, N_samples, B) bits in transmit order."""
        if self.CSI is None:
            raise ValueError("CSI must be updated before detection.")
        if self.possibleTXSymbols is None or self.bits is None:
            raise ValueError("Symbols and bit labels must be updated before detection.")
        received = torch.as_tensor(received, dtype=self.CSI.dtype, device=self.CSI.device)
        if received.ndim not in (1, 2) or received.shape[-1] != self.NR:
            raise ValueError("Received input must have shape (N_rx,) or (N_samples, N_rx).")
        return super().detectSequence(received).index_select(0, self._inverse_order)
