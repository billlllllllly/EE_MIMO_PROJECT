"""Inference driver for NVIDIA Sionna Research Kit's pretrained nrx_oai.onnx.

Get the model from NVIDIA's sionna-rk repository under
plugins/neural_receiver/models, then install numpy and onnxruntime.
The bundled model uses 24 PRBs, 16-QAM, and one transmit and receive antenna.
It expects 5G PUSCH slot data and pilot estimates, so the flat-fading
detectSequence interface in this project does not apply.
"""

import argparse
from pathlib import Path

import numpy as np


class NvidiaNRX:
    """Load the Sionna Research Kit ONNX export once and reuse it for slots."""

    _input_shapes = {
        "rx_slot": (1, 288, 13, 1, 2),
        "h_hat": (1, 432, 1, 1, 2),
        "active_dmrs_ports": (1, 1),
        "dmrs_ofdm_pos": (1, 3),
        "dmrs_subcarrier_pos": (1, 6),
    }

    def __init__(self, model_path):
        model_path = Path(model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"NVIDIA NRX model file not found: {model_path}")

        import onnxruntime as ort

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = {node.name for node in self._session.get_inputs()}
        outputs = [node.name for node in self._session.get_outputs()]
        if inputs != self._input_shapes.keys() or outputs != ["output_1"]:
            raise ValueError("Expected the Sionna Research Kit nrx_oai.onnx export.")

    def detect_llrs(
        self, *, rx_slot, h_hat, dmrs_ofdm_pos, dmrs_subcarrier_pos, active_dmrs_ports=((1,),)
    ):
        """Return the slot's coded-bit LLRs in the ONNX model's output layout.

        rx_slot is (1, 288, 13, 1, 2) for batch, subcarrier, OFDM symbol,
        receive antenna, and real/imaginary components. h_hat is
        (1, 432, 1, 1, 2) for batch, DMRS pilot (grouped by OFDM symbol,
        then by subcarrier), transmit port, receive antenna, and components.
        Both also accept complex arrays with the final component axis omitted.

        The active port mask is (1, 1). The pilot positions are integer arrays
        with shapes (1, 3) for OFDM symbols and (1, 6) for subcarrier offsets
        within a 12-subcarrier PRB. Use the positions from the input slot.
        The output preserves the full grid, including pilot positions.
        """
        values = {
            "rx_slot": rx_slot,
            "h_hat": h_hat,
            "active_dmrs_ports": active_dmrs_ports,
            "dmrs_ofdm_pos": dmrs_ofdm_pos,
            "dmrs_subcarrier_pos": dmrs_subcarrier_pos,
        }
        feeds = {}
        for name, value in values.items():
            array = np.asarray(value)
            if name in ("rx_slot", "h_hat") and np.iscomplexobj(array):
                array = np.stack((array.real, array.imag), axis=-1)
            shape = self._input_shapes[name]
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")

            if name in ("dmrs_ofdm_pos", "dmrs_subcarrier_pos"):
                limit = 13 if name == "dmrs_ofdm_pos" else 12
                if array.dtype.kind not in "iu" or np.any((array < 0) | (array >= limit)):
                    raise ValueError(f"{name} must contain integer indices in [0, {limit}).")
                feeds[name] = np.ascontiguousarray(array, dtype=np.int32)
            else:
                if array.dtype.kind not in "buif":
                    raise ValueError(f"{name} must contain real numbers after IQ packing.")
                feeds[name] = np.ascontiguousarray(array, dtype=np.float32)
        return self._session.run(["output_1"], feeds)[0]


def _main():
    parser = argparse.ArgumentParser(description="Run NVIDIA NRX on a synthetic 24-PRB slot.")
    parser.add_argument("model", help="Path to NVIDIA's nrx_oai.onnx")
    args = parser.parse_args()

    receiver = NvidiaNRX(args.model)
    rng = np.random.default_rng(0)
    llrs = receiver.detect_llrs(
        rx_slot=0.25 * rng.standard_normal((1, 288, 13, 1, 2)),
        h_hat=np.ones((1, 432, 1, 1), dtype=np.complex64),
        dmrs_ofdm_pos=[[2, 7, 11]],
        dmrs_subcarrier_pos=[[0, 2, 4, 6, 8, 10]],
    )
    if llrs.size != 288 * 13 * 4 or not np.isfinite(llrs).all():
        raise RuntimeError(f"Unexpected LLR output for a 24-PRB 16-QAM slot: {llrs.shape}.")
    print(f"NRX returned {llrs.size} finite LLRs with shape {llrs.shape}.")


if __name__ == "__main__":
    _main()
