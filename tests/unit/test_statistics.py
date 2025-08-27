"""Tests for statistics module - TDD RED phase."""

import numpy as np

from accelperm.core.statistics import (
    GLMStatistics,
    compute_f_statistics,
    compute_ols,
    compute_t_statistics,
)


class TestGLMStatistics:
    """Test the GLMStatistics class - RED phase."""

    def test_glm_statistics_exists(self):
        """Test that GLMStatistics class exists - RED phase."""
        # This should fail because GLMStatistics doesn't exist yet
        stats = GLMStatistics()
        assert isinstance(stats, GLMStatistics)

    def test_compute_ols_basic(self):
        """Test basic OLS computation - RED phase."""
        stats = GLMStatistics()

        # Simple linear regression: Y = X*beta + error
        n_subjects, n_regressors = 50, 2
        np.random.seed(42)
        X = np.random.randn(n_subjects, n_regressors)
        X[:, 0] = 1  # Intercept

        true_beta = np.array([2.0, 1.5])  # True coefficients
        Y = X @ true_beta + 0.1 * np.random.randn(n_subjects)

        result = stats.compute_ols(Y, X)

        # Check result structure
        assert "beta" in result
        assert "residuals" in result
        assert "mse" in result
        assert "df" in result

        # Check shapes
        assert result["beta"].shape == (n_regressors,)
        assert result["residuals"].shape == (n_subjects,)
        assert isinstance(result["mse"], float)
        assert result["df"] == n_subjects - n_regressors

        # Check numerical accuracy (should be close to true values)
        assert np.abs(result["beta"][0] - 2.0) < 0.5
        assert np.abs(result["beta"][1] - 1.5) < 0.5

    def test_compute_t_statistics_single_contrast(self):
        """Test t-statistic computation for single contrast - RED phase."""
        stats = GLMStatistics()

        # Create GLM results
        n_subjects, n_regressors = 30, 3
        beta = np.array([1.0, 2.0, -0.5])
        residuals = np.random.randn(n_subjects) * 0.5
        mse = np.mean(residuals**2)
        df = n_subjects - n_regressors

        # Design matrix (needed for standard error calculation)
        np.random.seed(123)
        X = np.random.randn(n_subjects, n_regressors)
        X[:, 0] = 1

        # Single contrast - test second coefficient
        contrast = np.array([0, 1, 0])

        result = stats.compute_t_statistics(beta, X, contrast, mse, df)

        # Check result structure
        assert "t_stat" in result
        assert "p_value" in result
        assert "se" in result

        # Check types
        assert isinstance(result["t_stat"], float)
        assert isinstance(result["p_value"], float)
        assert isinstance(result["se"], float)

        # Check reasonable values
        assert result["p_value"] >= 0.0 and result["p_value"] <= 1.0
        assert result["se"] > 0

    def test_compute_t_statistics_multiple_contrasts(self):
        """Test t-statistic computation for multiple contrasts - RED phase."""
        stats = GLMStatistics()

        n_subjects, n_regressors = 25, 3
        beta = np.array([1.0, 2.0, -1.0])
        residuals = np.random.randn(n_subjects) * 0.3
        mse = np.mean(residuals**2)
        df = n_subjects - n_regressors

        np.random.seed(456)
        X = np.random.randn(n_subjects, n_regressors)
        X[:, 0] = 1

        # Multiple contrasts
        contrasts = np.array(
            [
                [0, 1, 0],  # Test second coefficient
                [0, 0, 1],  # Test third coefficient
                [0, 1, -1],  # Difference between second and third
            ]
        )

        result = stats.compute_t_statistics(beta, X, contrasts, mse, df)

        # Check shapes for multiple contrasts
        assert result["t_stat"].shape == (3,)
        assert result["p_value"].shape == (3,)
        assert result["se"].shape == (3,)

        # All p-values should be valid
        assert np.all((result["p_value"] >= 0) & (result["p_value"] <= 1))

    def test_compute_f_statistics(self):
        """Test F-statistic computation - RED phase."""
        stats = GLMStatistics()

        n_subjects, n_regressors = 40, 4
        beta = np.array([1.0, 2.0, -1.0, 0.5])

        # Create residual sum of squares
        residuals = np.random.randn(n_subjects) * 0.4
        rss_full = np.sum(residuals**2)
        df_full = n_subjects - n_regressors

        # Reduced model (remove last two regressors)
        reduced_beta = beta[:2]
        reduced_residuals = np.random.randn(n_subjects) * 0.6  # Higher error
        rss_reduced = np.sum(reduced_residuals**2)
        df_reduced = n_subjects - 2

        result = stats.compute_f_statistics(rss_full, rss_reduced, df_full, df_reduced)

        # Check result structure
        assert "f_stat" in result
        assert "p_value" in result
        assert "df_num" in result
        assert "df_den" in result

        # Check types and values
        assert isinstance(result["f_stat"], float)
        assert isinstance(result["p_value"], float)
        assert result["f_stat"] >= 0
        assert 0 <= result["p_value"] <= 1
        assert result["df_num"] == df_reduced - df_full
        assert result["df_den"] == df_full


class TestUtilityFunctions:
    """Test standalone utility functions - RED phase."""

    def test_compute_ols_function_exists(self):
        """Test that compute_ols function exists - RED phase."""
        n_subjects, n_regressors = 20, 2

        np.random.seed(789)
        X = np.random.randn(n_subjects, n_regressors)
        X[:, 0] = 1
        Y = X @ np.array([1.5, 2.0]) + 0.2 * np.random.randn(n_subjects)

        result = compute_ols(Y, X)

        assert "beta" in result
        assert "residuals" in result
        assert result["beta"].shape == (n_regressors,)

    def test_compute_t_statistics_function_exists(self):
        """Test that compute_t_statistics function exists - RED phase."""
        beta = np.array([1.0, 2.0])
        n_subjects, n_regressors = 20, 2

        X = np.random.randn(n_subjects, n_regressors)
        X[:, 0] = 1
        contrast = np.array([0, 1])
        mse = 0.5
        df = n_subjects - n_regressors

        result = compute_t_statistics(beta, X, contrast, mse, df)

        assert "t_stat" in result
        assert isinstance(result["t_stat"], float)

    def test_compute_f_statistics_function_exists(self):
        """Test that compute_f_statistics function exists - RED phase."""
        rss_full = 10.5
        rss_reduced = 15.2
        df_full = 15
        df_reduced = 17

        result = compute_f_statistics(rss_full, rss_reduced, df_full, df_reduced)

        assert "f_stat" in result
        assert isinstance(result["f_stat"], float)


class TestStatisticsIntegration:
    """Test integration between statistics functions - RED phase."""

    def test_full_glm_pipeline(self):
        """Test complete GLM analysis pipeline - RED phase."""
        stats = GLMStatistics()

        # Create realistic neuroimaging scenario
        n_subjects, n_regressors = 30, 4
        np.random.seed(42)

        # Design matrix: intercept, age, group1, group2
        X = np.random.randn(n_subjects, n_regressors)
        X[:, 0] = 1  # Intercept
        X[:, 1] = np.random.uniform(18, 65, n_subjects)  # Age
        X[:, 2] = np.random.binomial(1, 0.5, n_subjects)  # Group 1
        X[:, 3] = 1 - X[:, 2]  # Group 2 (complementary)

        # True effects
        true_beta = np.array([50.0, 0.2, 5.0, -5.0])
        Y = X @ true_beta + np.random.randn(n_subjects) * 2.0

        # Step 1: Fit OLS
        ols_result = stats.compute_ols(Y, X)

        # Step 2: Test age effect
        age_contrast = np.array([0, 1, 0, 0])
        t_result = stats.compute_t_statistics(
            ols_result["beta"], X, age_contrast, ols_result["mse"], ols_result["df"]
        )

        # Step 3: Test overall group effect (F-test)
        # This would require fitting reduced model, but we'll test structure
        rss_full = ols_result["mse"] * ols_result["df"]
        rss_reduced = rss_full * 1.2  # Assume worse fit
        df_full = ols_result["df"]
        df_reduced = df_full + 2  # Removed 2 parameters

        f_result = stats.compute_f_statistics(
            rss_full, rss_reduced, df_full, df_reduced
        )

        # Verify pipeline worked
        assert ols_result["beta"].shape == (n_regressors,)
        assert isinstance(t_result["t_stat"], float)
        assert isinstance(f_result["f_stat"], float)

        # Age effect should be detectable (p < 0.05 likely)
        # Group effect should be highly significant
        assert t_result["p_value"] >= 0.0
        assert f_result["p_value"] >= 0.0

    def test_statistics_numerical_stability(self):
        """Test numerical stability with challenging data - RED phase."""
        stats = GLMStatistics()

        # Nearly collinear predictors (challenging for numerical stability)
        n_subjects = 100
        X = np.ones((n_subjects, 3))
        X[:, 1] = np.linspace(0, 1, n_subjects)
        X[:, 2] = X[:, 1] + 1e-10 * np.random.randn(n_subjects)  # Nearly identical

        Y = X[:, 1] * 2 + np.random.randn(n_subjects) * 0.1

        # OLS should handle this gracefully (using pseudoinverse)
        result = stats.compute_ols(Y, X)

        # Should not contain NaN or Inf
        assert np.all(np.isfinite(result["beta"]))
        assert np.isfinite(result["mse"])

        # Contrast testing should also be stable
        contrast = np.array([0, 1, 0])
        t_result = stats.compute_t_statistics(
            result["beta"], X, contrast, result["mse"], result["df"]
        )

        assert np.isfinite(t_result["t_stat"])
        assert np.isfinite(t_result["p_value"])
