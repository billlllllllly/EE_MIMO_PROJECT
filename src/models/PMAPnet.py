"""Neural regressors for the local bit LLRs in PMAP-Net."""

from collections.abc import Sequence

from torch import Tensor
from torch import nn


class PMAP_layer(nn.Module):
    """Fully connected network for one row of the QR-transformed system.

    For a row with J remaining streams and an M-point constellation, the
    input is the 2*(J*M+1) real features of Eq. (32). In order, the complex
    entries are z_k followed by each remaining R coefficient multiplied by
    every constellation point. Pack all real parts, then all imaginary parts.

    The output has J*log2(M) local bit LLRs, ordered by stream and then by bit,
    with the convention log(P(bit=0) / P(bit=1)).
    """

    def __init__(
        self,
        remaining_streams: int,
        modulation_order: int,
        *,
        hidden_layers: int,
        hidden_width_multiplier: int = 1,
    ) -> None:
        super().__init__()
        if remaining_streams < 1:
            raise ValueError("remaining_streams must be positive.")
        if modulation_order < 2 or modulation_order & (modulation_order - 1):
            raise ValueError("modulation_order must be a power of two of at least 2.")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive.")
        if hidden_width_multiplier < 1:
            raise ValueError("hidden_width_multiplier must be positive.")

        bits_per_symbol = modulation_order.bit_length() - 1
        base_width = remaining_streams * modulation_order + 1  # Table III
        hidden_size = base_width * hidden_width_multiplier
        self.input_size = 2 * base_width
        self.output_size = remaining_streams * bits_per_symbol

        layers: list[nn.Module] = [nn.Linear(self.input_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, self.output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class PMAPNet(nn.Module):
    """One independently parameterized regressor per QR row.

    Supply one prepared feature tensor per row, in order from the first row
    (all streams) to the last (one stream). Outputs follow the same order.
    The paper leaves the simulation depth unspecified, so hidden_layers is
    an explicit argument. hidden_width_multiplier scales the Table III widths.
    """

    def __init__(
        self,
        num_tx: int,
        modulation_order: int,
        *,
        hidden_layers: int,
        hidden_width_multiplier: int = 1,
    ) -> None:
        super().__init__()
        if num_tx < 1:
            raise ValueError("num_tx must be positive.")

        self.layer_networks = nn.ModuleList(
            PMAP_layer(
                remaining_streams,
                modulation_order,
                hidden_layers=hidden_layers,
                hidden_width_multiplier=hidden_width_multiplier,
            )
            for remaining_streams in range(num_tx, 0, -1)
        )

    def forward(self, features_by_row: Sequence[Tensor]) -> tuple[Tensor, ...]:
        if len(features_by_row) != len(self.layer_networks):
            raise ValueError("Supply one feature tensor per QR row.")
        return tuple(
            network(features)
            for network, features in zip(self.layer_networks, features_by_row)
        )
