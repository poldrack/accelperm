"""CPU backend implementation for AccelPerm."""


import numpy as np
from scipy import stats

from accelperm.backends.base import Backend


class CPUBackend(Backend):
    """CPU-based backend using NumPy and SciPy for GLM computations."""

    def __init__(self) -> None:
        """Initialize CPU backend."""
        self.name = "cpu"

    def is_available(self) -> bool:
        """Check if CPU backend is available (always True)."""
        return True

    def compute_glm(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        contrasts: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Compute General Linear Model statistics using CPU.

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
        dict[str, np.ndarray]
            Dictionary containing:
            - beta: Beta coefficients (n_voxels, n_regressors)
            - t_stat: T-statistics (n_voxels, n_contrasts)
            - p_values: P-values (n_voxels, n_contrasts)
            - residuals: Residuals (n_voxels, n_subjects)

        Raises
        ------
        ValueError
            If input dimensions are incompatible
        """
        # Validate input dimensions
        self._validate_inputs(Y, X, contrasts)

        n_voxels, n_subjects = Y.shape
        n_regressors = X.shape[1]
        n_contrasts = contrasts.shape[0]

        # Compute GLM: beta = (X'X)^(-1) X'Y
        # For Y: (n_voxels, n_subjects), X: (n_subjects, n_regressors)
        # We need to solve for each voxel: Y_voxel = X @ beta_voxel
        # This means beta = (X'X)^(-1) X' Y'  where Y' is (n_subjects, n_voxels)
        XtX_inv = np.linalg.pinv(X.T @ X)  # Shape: (n_regressors, n_regressors)
        XtY = (
            X.T @ Y.T
        )  # X.T: (n_regressors, n_subjects), Y.T: (n_subjects, n_voxels) -> (n_regressors, n_voxels)
        beta = XtX_inv @ XtY  # Shape: (n_regressors, n_voxels)
        beta = beta.T  # Transpose to (n_voxels, n_regressors)

        # Compute residuals
        Y_pred = (
            X @ beta.T
        ).T  # X @ beta.T gives (n_subjects, n_voxels), transpose to (n_voxels, n_subjects)
        residuals = Y - Y_pred  # Shape: (n_voxels, n_subjects)

        # Compute residual sum of squares and degrees of freedom
        rss = np.sum(residuals**2, axis=1)  # Shape: (n_voxels,)
        df = n_subjects - n_regressors

        # Handle edge case where df = 0 (no degrees of freedom)
        mse = np.full_like(rss, np.inf) if df <= 0 else rss / df  # Mean squared error

        # Compute standard errors and t-statistics for each contrast
        t_stats = np.zeros((n_voxels, n_contrasts))
        p_values = np.zeros((n_voxels, n_contrasts))

        for i, contrast in enumerate(contrasts):
            # Compute contrast effect: c'�
            contrast_effect = beta @ contrast  # Shape: (n_voxels,)

            # Compute standard error: sqrt(c' (X'X)^(-1) c * MSE)
            contrast_var = contrast.T @ XtX_inv @ contrast  # Scalar
            se = np.sqrt(contrast_var * mse)  # Shape: (n_voxels,)

            # Compute t-statistic
            # Handle division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                t_stat = contrast_effect / se
                t_stat = np.where(se == 0, 0, t_stat)  # Set to 0 where se is 0

            t_stats[:, i] = t_stat

            # Compute p-values (two-tailed)
            p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
            p_values[:, i] = p_vals

        return {
            "beta": beta,
            "t_stat": t_stats,
            "p_values": p_values,
            "residuals": residuals,
        }

    def _validate_inputs(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        contrasts: np.ndarray,
    ) -> None:
        """Validate input dimensions and compatibility."""
        # Check that Y and X have compatible dimensions
        if Y.shape[1] != X.shape[0]:
            raise ValueError(
                f"Dimension mismatch: Y has {Y.shape[1]} subjects, "
                f"X has {X.shape[0]} subjects"
            )

        # Check that contrasts and X have compatible dimensions
        if contrasts.shape[1] != X.shape[1]:
            raise ValueError(
                f"Contrast dimension mismatch: contrasts have {contrasts.shape[1]} "
                f"regressors, X has {X.shape[1]} regressors"
            )

        # Check for valid shapes
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D array, got {Y.ndim}D")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if contrasts.ndim != 2:
            raise ValueError(f"Contrasts must be 2D array, got {contrasts.ndim}D")
