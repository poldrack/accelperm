"""Backend abstraction layer for GPU-accelerated permutation testing."""

from abc import ABC, abstractmethod

import numpy as np
import torch


class Backend(ABC):
    """Abstract base class for all computational backends."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        pass

    @abstractmethod
    def compute_glm(
        self,
        Y: np.ndarray | torch.Tensor,
        X: np.ndarray | torch.Tensor,
        contrasts: np.ndarray | torch.Tensor,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Compute General Linear Model statistics."""
        pass
