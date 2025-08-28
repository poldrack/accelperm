"""Data chunking system for handling large datasets."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import psutil

from accelperm.backends.base import Backend


class DataChunker:
    """Utility class for chunking large neuroimaging datasets."""

    def __init__(self) -> None:
        """Initialize data chunker."""
        pass

    def calculate_optimal_chunk_size(
        self, data_shape: tuple[int, int, int], available_memory_gb: float
    ) -> int:
        """
        Calculate optimal chunk size based on data shape and available memory.

        Parameters
        ----------
        data_shape : tuple[int, int, int]
            Data dimensions (n_voxels, n_subjects, n_regressors)
        available_memory_gb : float
            Available memory in GB

        Returns
        -------
        int
            Optimal chunk size (number of voxels per chunk)
        """
        n_voxels, n_subjects, n_regressors = data_shape

        # Estimate memory per voxel (in bytes)
        # Y: n_subjects * 8 bytes (float64)
        # Beta: n_regressors * 8 bytes
        # Residuals: n_subjects * 8 bytes
        # T-stats, p-values: n_contrasts * 8 bytes (assume avg 2 contrasts)
        # Plus intermediate calculations (safety factor of 3x)
        memory_per_voxel = (n_subjects + n_regressors + n_subjects + 2) * 8 * 3

        # Convert available memory to bytes
        available_bytes = available_memory_gb * 1024**3

        # Calculate max voxels that fit in memory
        max_voxels_per_chunk = int(
            available_bytes * 0.8 / memory_per_voxel
        )  # 80% safety

        # Ensure at least 1 voxel per chunk
        max_voxels_per_chunk = max(1, max_voxels_per_chunk)

        # Don't chunk if data is small enough
        if max_voxels_per_chunk >= n_voxels:
            return n_voxels

        # Ensure we don't return the full data size if memory is constrained
        return min(max_voxels_per_chunk, n_voxels - 1)

    def chunk_data(
        self, Y: np.ndarray, X: np.ndarray, chunk_size: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray, int, int]]:
        """
        Generate data chunks for processing.

        Parameters
        ----------
        Y : np.ndarray
            Data matrix (n_voxels, n_subjects)
        X : np.ndarray
            Design matrix (n_subjects, n_regressors)
        chunk_size : int
            Number of voxels per chunk

        Yields
        ------
        tuple[np.ndarray, np.ndarray, int, int]
            (chunk_Y, chunk_X, start_idx, end_idx)
        """
        n_voxels = Y.shape[0]

        for start_idx in range(0, n_voxels, chunk_size):
            end_idx = min(start_idx + chunk_size, n_voxels)
            chunk_Y = Y[start_idx:end_idx]

            yield chunk_Y, X, start_idx, end_idx

    def reconstruct_results(
        self, chunk_results: list[dict[str, np.ndarray]], total_voxels: int
    ) -> dict[str, np.ndarray]:
        """
        Reconstruct full results from chunk results.

        Parameters
        ----------
        chunk_results : list[dict[str, np.ndarray]]
            List of GLM results from each chunk
        total_voxels : int
            Total number of voxels

        Returns
        -------
        dict[str, np.ndarray]
            Reconstructed full results
        """
        if not chunk_results:
            raise ValueError("No chunk results provided")

        # Get dimensions from first chunk
        first_result = chunk_results[0]
        n_regressors = first_result["beta"].shape[1]
        n_contrasts = first_result["t_stat"].shape[1]

        # Initialize full result arrays
        full_beta = np.zeros((total_voxels, n_regressors))
        full_t_stat = np.zeros((total_voxels, n_contrasts))
        full_p_values = np.zeros((total_voxels, n_contrasts))

        # Reconstruct by concatenating chunks
        current_idx = 0
        for chunk_result in chunk_results:
            chunk_size = chunk_result["beta"].shape[0]
            end_idx = current_idx + chunk_size

            full_beta[current_idx:end_idx] = chunk_result["beta"]
            full_t_stat[current_idx:end_idx] = chunk_result["t_stat"]
            full_p_values[current_idx:end_idx] = chunk_result["p_values"]

            current_idx = end_idx

        return {
            "beta": full_beta,
            "t_stat": full_t_stat,
            "p_values": full_p_values,
        }


class ChunkedBackendWrapper:
    """Wrapper that adds chunking capability to any backend."""

    def __init__(
        self,
        backend: Backend,
        chunk_size: int | None = None,
        max_memory_gb: float | None = None,
    ) -> None:
        """
        Initialize chunked backend wrapper.

        Parameters
        ----------
        backend : Backend
            Base backend to wrap
        chunk_size : int | None
            Fixed chunk size, or None for auto-calculation
        max_memory_gb : float | None
            Maximum memory to use, or None for auto-detection
        """
        self.backend = backend
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.chunker = DataChunker()

        # Auto-detect memory if not specified
        if self.max_memory_gb is None:
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            self.max_memory_gb = total_memory_gb * 0.5  # Use 50% of available memory

    @property
    def name(self) -> str:
        """Backend name."""
        return f"chunked_{self.backend.name}"

    def is_available(self) -> bool:
        """Check if backend is available."""
        return self.backend.is_available()

    def compute_glm(
        self, Y: np.ndarray, X: np.ndarray, contrasts: np.ndarray
    ) -> dict[str, Any]:
        """
        Compute GLM using chunked processing.

        Parameters
        ----------
        Y : np.ndarray
            Data matrix (n_voxels, n_subjects)
        X : np.ndarray
            Design matrix (n_subjects, n_regressors)
        contrasts : np.ndarray
            Contrast matrix (n_contrasts, n_regressors)

        Returns
        -------
        dict[str, Any]
            GLM results combined from all chunks
        """
        n_voxels = Y.shape[0]

        # Calculate chunk size if not fixed
        if self.chunk_size is None:
            data_shape = (n_voxels, Y.shape[1], X.shape[1])
            chunk_size = self.chunker.calculate_optimal_chunk_size(
                data_shape, self.max_memory_gb
            )
        else:
            chunk_size = self.chunk_size

        # If chunk size covers all data, just use backend directly
        if chunk_size >= n_voxels:
            return self.backend.compute_glm(Y, X, contrasts)

        # Process in chunks
        chunk_results = []
        for chunk_Y, chunk_X, _start_idx, _end_idx in self.chunker.chunk_data(
            Y, X, chunk_size
        ):
            chunk_result = self.backend.compute_glm(chunk_Y, chunk_X, contrasts)
            chunk_results.append(chunk_result)

        # Reconstruct full results
        return self.chunker.reconstruct_results(chunk_results, n_voxels)

    def apply_permutation(self, data: np.ndarray, strategy: str) -> np.ndarray:
        """Apply permutation strategy to data."""
        return self.backend.apply_permutation(data, strategy)
