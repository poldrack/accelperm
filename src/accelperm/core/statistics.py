"""Statistical computations for AccelPerm GLM analysis."""

from typing import Any

import numpy as np
from scipy import stats


class GLMStatistics:
    """Class for computing General Linear Model statistics."""

    def __init__(self) -> None:
        """Initialize GLMStatistics."""
        self.name = "glm_statistics"

    def compute_ols(self, Y: np.ndarray, X: np.ndarray) -> dict[str, Any]:
        """
        Compute Ordinary Least Squares regression.

        Parameters
        ----------
        Y : np.ndarray
            Response vector of shape (n_subjects,)
        X : np.ndarray
            Design matrix of shape (n_subjects, n_regressors)

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - beta: Regression coefficients (n_regressors,)
            - residuals: Residuals (n_subjects,)
            - mse: Mean squared error (scalar)
            - df: Degrees of freedom (scalar)
        """
        n_subjects, n_regressors = X.shape

        # Compute beta coefficients: beta = (X'X)^(-1) X'Y
        XtX_inv = np.linalg.pinv(X.T @ X)  # Use pseudoinverse for stability
        beta = XtX_inv @ (X.T @ Y)

        # Compute residuals
        Y_pred = X @ beta
        residuals = Y - Y_pred

        # Compute mean squared error
        df = n_subjects - n_regressors
        rss = np.sum(residuals**2)
        mse = rss / df if df > 0 else np.inf

        return {
            "beta": beta,
            "residuals": residuals,
            "mse": mse,
            "df": df,
        }

    def compute_t_statistics(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        contrast: np.ndarray,
        mse: float,
        df: int,
    ) -> dict[str, Any]:
        """
        Compute t-statistics for given contrast(s).

        Parameters
        ----------
        beta : np.ndarray
            Regression coefficients (n_regressors,)
        X : np.ndarray
            Design matrix (n_subjects, n_regressors)
        contrast : np.ndarray
            Contrast vector (n_regressors,) or matrix (n_contrasts, n_regressors)
        mse : float
            Mean squared error from GLM fit
        df : int
            Degrees of freedom

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - t_stat: T-statistic(s)
            - p_value: P-value(s) (two-tailed)
            - se: Standard error(s)
        """
        # Handle both single contrast and multiple contrasts
        if contrast.ndim == 1:
            contrast = contrast.reshape(1, -1)
            single_contrast = True
        else:
            single_contrast = False

        n_contrasts = contrast.shape[0]
        XtX_inv = np.linalg.pinv(X.T @ X)

        t_stats = np.zeros(n_contrasts)
        p_values = np.zeros(n_contrasts)
        standard_errors = np.zeros(n_contrasts)

        for i, c in enumerate(contrast):
            # Compute contrast effect: c'²
            contrast_effect = c @ beta

            # Compute standard error: sqrt(c' (X'X)^(-1) c * MSE)
            contrast_var = c @ XtX_inv @ c.T
            se = np.sqrt(contrast_var * mse)
            standard_errors[i] = se

            # Compute t-statistic
            if se > 0:
                t_stat = contrast_effect / se
            else:
                t_stat = 0.0
            t_stats[i] = t_stat

            # Compute two-tailed p-value
            if df > 0:
                p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
            else:
                p_val = 1.0
            p_values[i] = p_val

        # Return scalars for single contrast, arrays for multiple
        if single_contrast:
            return {
                "t_stat": t_stats[0],
                "p_value": p_values[0],
                "se": standard_errors[0],
            }
        else:
            return {
                "t_stat": t_stats,
                "p_value": p_values,
                "se": standard_errors,
            }

    def compute_f_statistics(
        self,
        rss_full: float,
        rss_reduced: float,
        df_full: int,
        df_reduced: int,
    ) -> dict[str, Any]:
        """
        Compute F-statistic for model comparison.

        Parameters
        ----------
        rss_full : float
            Residual sum of squares for full model
        rss_reduced : float
            Residual sum of squares for reduced model
        df_full : int
            Degrees of freedom for full model
        df_reduced : int
            Degrees of freedom for reduced model

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - f_stat: F-statistic
            - p_value: P-value
            - df_num: Numerator degrees of freedom
            - df_den: Denominator degrees of freedom
        """
        # Degrees of freedom
        df_num = df_reduced - df_full  # Difference in parameters
        df_den = df_full

        # F-statistic calculation
        if df_num > 0 and df_den > 0 and rss_full > 0:
            f_stat = ((rss_reduced - rss_full) / df_num) / (rss_full / df_den)
            p_value = 1 - stats.f.cdf(f_stat, df_num, df_den)
        else:
            f_stat = 0.0
            p_value = 1.0

        return {
            "f_stat": f_stat,
            "p_value": p_value,
            "df_num": df_num,
            "df_den": df_den,
        }


# Standalone utility functions
def compute_ols(Y: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    """
    Compute Ordinary Least Squares regression.

    Parameters
    ----------
    Y : np.ndarray
        Response vector of shape (n_subjects,)
    X : np.ndarray
        Design matrix of shape (n_subjects, n_regressors)

    Returns
    -------
    dict[str, Any]
        Dictionary containing beta, residuals, mse, df
    """
    stats_obj = GLMStatistics()
    return stats_obj.compute_ols(Y, X)


def compute_t_statistics(
    beta: np.ndarray,
    X: np.ndarray,
    contrast: np.ndarray,
    mse: float,
    df: int,
) -> dict[str, Any]:
    """
    Compute t-statistics for given contrast(s).

    Parameters
    ----------
    beta : np.ndarray
        Regression coefficients
    X : np.ndarray
        Design matrix
    contrast : np.ndarray
        Contrast vector or matrix
    mse : float
        Mean squared error
    df : int
        Degrees of freedom

    Returns
    -------
    dict[str, Any]
        Dictionary containing t_stat, p_value, se
    """
    stats_obj = GLMStatistics()
    return stats_obj.compute_t_statistics(beta, X, contrast, mse, df)


def compute_f_statistics(
    rss_full: float,
    rss_reduced: float,
    df_full: int,
    df_reduced: int,
) -> dict[str, Any]:
    """
    Compute F-statistic for model comparison.

    Parameters
    ----------
    rss_full : float
        Residual sum of squares for full model
    rss_reduced : float
        Residual sum of squares for reduced model
    df_full : int
        Degrees of freedom for full model
    df_reduced : int
        Degrees of freedom for reduced model

    Returns
    -------
    dict[str, Any]
        Dictionary containing f_stat, p_value, df_num, df_den
    """
    stats_obj = GLMStatistics()
    return stats_obj.compute_f_statistics(rss_full, rss_reduced, df_full, df_reduced)
