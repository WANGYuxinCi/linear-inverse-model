"""linear_inverse_model (LIM) toolbox."""

__all__ = ["CSLIM", "STLIM", "LIM_utils", "__version__"]
__version__ = "0.1.0"

from .CSLIM import CSLIM
from .STLIM import STLIM
from . import LIM_utils
