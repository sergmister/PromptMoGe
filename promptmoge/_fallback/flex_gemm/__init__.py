"""Pure-PyTorch stand-in for FlexGEMM (macOS / MPS). See _core.py."""
from . import ops, nn
from ._core import STATS, QUANT

__version__ = "0.0.1+torch-reference"
__all__ = ["ops", "nn", "STATS", "QUANT"]
