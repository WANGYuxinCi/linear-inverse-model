"""linear_inverse_model (LIM) toolbox."""

__version__ = "0.1.1"

from .CSLIM import CSLIM
from .STLIM import STLIM
from . import LIM_utils

__all__ = ["CSLIM", "STLIM", "LIM_utils", "__version__"]
