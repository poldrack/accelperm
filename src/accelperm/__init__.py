"""AccelPerm: GPU-accelerated permutation testing for neuroimaging data."""

__version__ = "0.1.0"
__author__ = "Russell Poldrack"
__email__ = "poldrack@gmail.com"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from accelperm.backends.base import Backend
    from accelperm.core.orchestrator import Orchestrator

__all__ = ["__version__", "Backend", "Orchestrator"]
