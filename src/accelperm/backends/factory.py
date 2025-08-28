"""Backend factory for intelligent backend selection."""

import threading
from typing import Any

import psutil

from accelperm.backends.base import Backend
from accelperm.backends.cpu import CPUBackend


class BackendFactory:
    """Factory for creating and managing backend instances."""

    def __init__(self) -> None:
        """Initialize backend factory."""
        self._backends: dict[str, Backend] = {}
        self._lock = threading.Lock()

    def get_best_backend(self) -> Backend:
        """
        Get the best available backend automatically.

        Returns
        -------
        Backend
            The optimal backend for the current system
        """
        # Check MPS availability first (prefer GPU when available)
        try:
            import torch

            if torch.backends.mps.is_available():
                return self.get_backend("mps")
        except ImportError:
            pass

        # Fallback to CPU
        return self.get_backend("cpu")

    def get_backend(
        self,
        backend_name: str,
        enable_chunking: bool = False,
        max_memory_gb: float | None = None,
    ) -> Backend:
        """
        Get a specific backend by name.

        Parameters
        ----------
        backend_name : str
            Name of the backend ("cpu", "mps")
        enable_chunking : bool, default=False
            Whether to wrap the backend with chunking capability
        max_memory_gb : float, optional
            Maximum memory to use for chunked processing

        Returns
        -------
        Backend
            The requested backend instance, optionally with chunking

        Raises
        ------
        ValueError
            If backend name is invalid
        RuntimeError
            If backend is not available on this system
        """
        backend_name = backend_name.lower()

        # Validate backend name
        valid_backends = ["cpu", "mps"]
        if backend_name not in valid_backends:
            raise ValueError(
                f"Invalid backend '{backend_name}'. "
                f"Valid options: {', '.join(valid_backends)}"
            )

        # Thread-safe backend creation
        with self._lock:
            if backend_name not in self._backends:
                self._backends[backend_name] = self._create_backend(backend_name)

        backend = self._backends[backend_name]

        # Verify backend is available
        if not backend.is_available():
            raise RuntimeError(
                f"Backend '{backend_name}' is not available on this system"
            )

        # Wrap with chunking if requested
        if enable_chunking:
            from accelperm.core.chunking import ChunkedBackendWrapper

            backend = ChunkedBackendWrapper(
                backend=backend, max_memory_gb=max_memory_gb
            )

        return backend

    def _create_backend(self, backend_name: str) -> Backend:
        """Create a new backend instance."""
        if backend_name == "cpu":
            return CPUBackend()
        elif backend_name == "mps":
            from accelperm.backends.mps import MPSBackend

            return MPSBackend()
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    def get_optimal_backend(self, data_shape: tuple[int, int, int]) -> Backend:
        """
        Choose optimal backend based on data characteristics.

        Parameters
        ----------
        data_shape : tuple[int, int, int]
            Data dimensions (n_voxels, n_subjects, n_regressors)

        Returns
        -------
        Backend
            The optimal backend for this data size
        """
        n_voxels, n_subjects, n_regressors = data_shape

        # Estimate computational complexity
        flops = n_voxels * n_subjects * n_regressors * 10  # Rough estimate
        memory_mb = self.estimate_memory_requirements(data_shape)

        # Small datasets: CPU is often faster due to GPU overhead
        if flops < 1e6 or memory_mb < 10:
            return self.get_backend("cpu")

        # Large datasets: prefer GPU if available
        try:
            return self.get_backend("mps")
        except (ValueError, RuntimeError):
            return self.get_backend("cpu")

    def estimate_memory_requirements(self, data_shape: tuple[int, int, int]) -> float:
        """
        Estimate memory requirements for given data shape.

        Parameters
        ----------
        data_shape : tuple[int, int, int]
            Data dimensions (n_voxels, n_subjects, n_regressors)

        Returns
        -------
        float
            Estimated memory requirement in MB
        """
        n_voxels, n_subjects, n_regressors = data_shape

        # Memory for data matrices (float32/64)
        data_memory = n_voxels * n_subjects * 8  # Y matrix (float64)
        design_memory = n_subjects * n_regressors * 8  # X matrix

        # Memory for intermediate computations
        # XtX: n_regressors^2, XtY: n_regressors * n_voxels
        intermediate_memory = (
            n_regressors * n_regressors * 8  # XtX
            + n_regressors * n_voxels * 8  # XtY
            + n_voxels * n_regressors * 8  # beta
            + n_voxels * n_subjects * 8  # residuals
            + n_voxels * 8  # t_stats, p_values per contrast
        )

        total_bytes = (
            data_memory + design_memory + intermediate_memory * 2
        )  # Safety factor
        return total_bytes / (1024 * 1024)  # Convert to MB

    def list_available_backends(self) -> list[str]:
        """
        List all available backends on this system.

        Returns
        -------
        list[str]
            Names of available backends
        """
        available = []

        # CPU is always available
        available.append("cpu")

        # Check MPS availability
        try:
            import torch

            if torch.backends.mps.is_available():
                available.append("mps")
        except ImportError:
            pass

        return available

    def get_backend_capabilities(self, backend_name: str) -> dict[str, Any]:
        """
        Get capabilities information for a backend.

        Parameters
        ----------
        backend_name : str
            Name of the backend

        Returns
        -------
        dict[str, Any]
            Backend capabilities information
        """
        capabilities = {}

        if backend_name == "cpu":
            # CPU capabilities
            capabilities.update(
                {
                    "max_memory_gb": psutil.virtual_memory().total / (1024**3),
                    "supports_float64": True,
                    "supports_float32": True,
                    "device_type": "cpu",
                    "cores": psutil.cpu_count(),
                    "parallel_processing": True,
                }
            )

        elif backend_name == "mps":
            # MPS capabilities
            try:
                import torch

                if torch.backends.mps.is_available():
                    # Get system memory as proxy for GPU memory (MPS uses unified memory)
                    total_memory_gb = psutil.virtual_memory().total / (1024**3)
                    capabilities.update(
                        {
                            "max_memory_gb": total_memory_gb
                            * 0.7,  # Conservative estimate
                            "supports_float64": False,  # MPS limitation
                            "supports_float32": True,
                            "device_type": "mps",
                            "cores": "unified",
                            "parallel_processing": True,
                            "gpu_acceleration": True,
                        }
                    )
                else:
                    capabilities.update(
                        {
                            "available": False,
                            "reason": "MPS not available on this system",
                        }
                    )
            except ImportError:
                capabilities.update(
                    {"available": False, "reason": "PyTorch not available"}
                )
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

        return capabilities
