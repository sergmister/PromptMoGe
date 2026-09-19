"""Pure-PyTorch stand-in for FlexGEMM (CPU / Apple silicon). See _core.py."""
from . import ops, nn

__version__ = "0.0.1+torch-reference"
__all__ = ["ops", "nn"]
