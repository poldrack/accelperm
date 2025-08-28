"""MPS (Metal Performance Shaders) backend implementation for AccelPerm."""

from typing import Any

import numpy as np
import torch
from scipy import stats

from accelperm.backends.base import Backend


class MPSBackend(Backend):
    """MPS backend for GPU-accelerated GLM computation on Apple Silicon."""

    def __init__(self) -> None:
        """Initialize MPS backend with device management."""
        self.name = "mps"

        # Initialize device - fallback to CPU if MPS unavailable
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def is_available(self) -> bool:
        """Check if MPS backend is available."""
        return torch.backends.mps.is_available()

    def compute_glm(
        self, Y: np.ndarray, X: np.ndarray, contrasts: np.ndarray
    ) -> dict[str, Any]:
        """
        Compute GLM statistics using MPS acceleration.

        Parameters
        ----------
        Y : np.ndarray
            Data matrix of shape (n_voxels, n_subjects)
        X : np.ndarray
            Design matrix of shape (n_subjects, n_regressors)
        contrasts : np.ndarray
            Contrast matrix of shape (n_contrasts, n_regressors)

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - beta: Regression coefficients (n_voxels, n_regressors)
            - t_stat: T-statistics (n_voxels, n_contrasts)
            - p_values: P-values (n_voxels, n_contrasts)
        """
        # Validate inputs
        self._validate_inputs(Y, X, contrasts)

        try:
            # Convert to PyTorch tensors and move to device
            Y_tensor = self._numpy_to_tensor(Y)
            X_tensor = self._numpy_to_tensor(X)
            contrasts_tensor = self._numpy_to_tensor(contrasts)

            # Compute GLM using tensor operations
            result = self._compute_glm_tensors(Y_tensor, X_tensor, contrasts_tensor)

            # Convert results back to NumPy
            beta = self._tensor_to_numpy(result["beta"])
            t_stat = self._tensor_to_numpy(result["t_stat"])
            p_values = self._tensor_to_numpy(result["p_values"])

            # Cleanup GPU memory
            self._cleanup_memory()

            return {"beta": beta, "t_stat": t_stat, "p_values": p_values}

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._handle_out_of_memory()
                # Fallback to CPU computation
                return self._compute_glm_cpu_fallback(Y, X, contrasts)
            else:
                raise

    def _compute_glm_tensors(
        self, Y: torch.Tensor, X: torch.Tensor, contrasts: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute GLM using PyTorch tensor operations."""
        n_voxels, n_subjects = Y.shape
        n_regressors = X.shape[1]
        n_contrasts = contrasts.shape[0]

        # Compute GLM: beta = (X'X)^(-1) X'Y
        # For Y: (n_voxels, n_subjects), X: (n_subjects, n_regressors)
        XtX = X.T @ X  # (n_regressors, n_regressors)

        # Use MPS-compatible matrix inversion
        # Add regularization for numerical stability
        reg_factor = 1e-8
        XtX_reg = XtX + reg_factor * torch.eye(
            XtX.shape[0], device=self.device, dtype=XtX.dtype
        )

        try:
            # Try Cholesky decomposition (faster for positive definite matrices)
            L = torch.linalg.cholesky(XtX_reg)
            XtX_inv = torch.cholesky_inverse(L)
            XtY = X.T @ Y.T  # (n_regressors, n_voxels)
            beta = (XtX_inv @ XtY).T  # (n_voxels, n_regressors)
        except RuntimeError:
            # Fall back to QR-based solve if Cholesky fails
            XtY = X.T @ Y.T  # (n_regressors, n_voxels)
            beta = torch.linalg.solve(XtX_reg, XtY).T  # (n_voxels, n_regressors)
            # Still need the inverse for variance calculations
            try:
                XtX_inv = torch.linalg.inv(XtX_reg)
            except RuntimeError:
                # If inversion fails, use solve for each contrast individually
                XtX_inv = None

        # Compute residuals
        Y_pred = (X @ beta.T).T  # (n_voxels, n_subjects)
        residuals = Y - Y_pred

        # Compute mean squared error
        rss = torch.sum(residuals**2, dim=1)  # (n_voxels,)
        df = n_subjects - n_regressors

        # Handle edge case where df = 0
        if df <= 0:
            mse = torch.full_like(rss, float("inf"))
        else:
            mse = rss / df

        # Compute t-statistics for each contrast
        t_stats = torch.zeros(n_voxels, n_contrasts, device=self.device)
        p_values = torch.zeros(n_voxels, n_contrasts, device=self.device)

        for i, contrast in enumerate(contrasts):
            # Contrast effect: c'�
            contrast_effects = beta @ contrast  # (n_voxels,)

            # Standard error: sqrt(c' (X'X)^(-1) c * MSE)
            if XtX_inv is not None:
                contrast_var = contrast @ XtX_inv @ contrast  # scalar
            else:
                # Solve (X'X) v = c for variance calculation if inverse not available
                v = torch.linalg.solve(XtX_reg, contrast)
                contrast_var = contrast @ v

            se = torch.sqrt(contrast_var * mse)  # (n_voxels,)

            # T-statistic
            t_stat = torch.where(
                se > 0, contrast_effects / se, torch.tensor(0.0, device=self.device)
            )
            t_stats[:, i] = t_stat

            # P-values (using CPU for scipy.stats)
            if df > 0:
                t_stat_cpu = t_stat.cpu().numpy()
                p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stat_cpu), df))
                p_values[:, i] = torch.from_numpy(p_vals.astype(np.float32)).to(
                    self.device
                )
            else:
                p_values[:, i] = 1.0

        return {"beta": beta, "t_stat": t_stats, "p_values": p_values}

    def _compute_glm_cpu_fallback(
        self, Y: np.ndarray, X: np.ndarray, contrasts: np.ndarray
    ) -> dict[str, Any]:
        """Fallback to CPU computation when MPS fails."""
        from accelperm.backends.cpu import CPUBackend

        cpu_backend = CPUBackend()
        return cpu_backend.compute_glm(Y, X, contrasts)

    def _validate_inputs(
        self, Y: np.ndarray, X: np.ndarray, contrasts: np.ndarray
    ) -> None:
        """Validate input arrays for GLM computation."""
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D array, got {Y.ndim}D")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if contrasts.ndim != 2:
            raise ValueError(f"Contrasts must be 2D array, got {contrasts.ndim}D")

        if Y.shape[1] != X.shape[0]:
            raise ValueError(
                f"Dimension mismatch: Y has {Y.shape[1]} subjects, "
                f"X has {X.shape[0]} subjects"
            )

        if X.shape[1] != contrasts.shape[1]:
            raise ValueError(
                f"Dimension mismatch: X has {X.shape[1]} regressors, "
                f"contrasts have {contrasts.shape[1]} regressors"
            )

    def _numpy_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert NumPy array to PyTorch tensor on device."""
        # Ensure float32 for MPS compatibility
        tensor = torch.from_numpy(array.astype(np.float32))
        return self._to_device(tensor)

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor back to NumPy array."""
        return tensor.cpu().numpy()

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the appropriate device."""
        return tensor.to(self.device)

    def _cleanup_memory(self) -> None:
        """Clean up GPU memory."""
        if self.device.type == "mps":
            torch.mps.empty_cache()

    def _handle_out_of_memory(self) -> None:
        """Handle out-of-memory errors."""
        self._cleanup_memory()
        # Could implement additional recovery strategies here

    def apply_permutation(self, data: np.ndarray, strategy: str) -> np.ndarray:
        """
        Apply permutation strategy to data.

        Parameters
        ----------
        data : np.ndarray
            Input data to permute
        strategy : str
            Permutation strategy ("sign_flip", "full", "monte_carlo")

        Returns
        -------
        np.ndarray
            Permuted data
        """
        # Placeholder implementation - will be expanded in future weeks
        if strategy == "sign_flip":
            # Simple sign flipping for now
            signs = np.random.choice([-1, 1], size=data.shape[1])
            return data * signs
        else:
            raise NotImplementedError(
                f"Permutation strategy '{strategy}' not implemented yet"
            )
