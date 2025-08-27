"""AccelPerm: GPU-accelerated permutation testing for neuroimaging data."""

__version__ = "0.1.0"
__author__ = "Russell Poldrack"
__email__ = "poldrack@gmail.com"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from accelperm.backends.base import Backend
    from accelperm.config import (
        AccelPermConfig,
        BackendConfig,
        ComputeConfig,
        LoggingConfig,
        OutputConfig,
        ValidationLevel,
        load_config,
        save_config,
    )
    from accelperm.core.orchestrator import Orchestrator
    from accelperm.exceptions import (
        AccelPermError,
        BackendError,
        ConfigurationError,
        DataError,
        InsufficientMemoryError,
    )
    from accelperm.utils.logging import (
        AccelPermLogger,
        LogLevel,
        ProgressReporter,
        setup_logger,
    )

__all__ = [
    "__version__",
    "Backend",
    "Orchestrator",
    "AccelPermError",
    "BackendError",
    "ConfigurationError",
    "DataError",
    "InsufficientMemoryError",
    "AccelPermLogger",
    "LogLevel",
    "ProgressReporter",
    "setup_logger",
    "AccelPermConfig",
    "BackendConfig",
    "ComputeConfig",
    "LoggingConfig",
    "OutputConfig",
    "ValidationLevel",
    "load_config",
    "save_config",
]
