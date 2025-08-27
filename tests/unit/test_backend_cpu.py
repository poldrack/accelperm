"""Tests for CPU backend - TDD RED phase."""

import numpy as np
import pytest

from accelperm.backends.cpu import CPUBackend


class TestCPUBackend:
    """Test the CPUBackend class - RED phase."""

    def test_cpu_backend_exists(self):
        """Test that CPUBackend class exists - RED phase."""
        # This should fail because CPUBackend doesn't exist yet
        backend = CPUBackend()
        assert isinstance(backend, CPUBackend)

    def test_cpu_backend_is_available(self):
        """Test that CPU backend reports as available - RED phase."""
        # CPU should always be available
        backend = CPUBackend()
        assert backend.is_available() is True

    def test_cpu_backend_inherits_from_backend(self):
        """Test that CPUBackend inherits from Backend ABC - RED phase."""
        from accelperm.backends.base import Backend

        backend = CPUBackend()
        assert isinstance(backend, Backend)

    def test_compute_glm_with_simple_data(self):
        """Test basic GLM computation with simple synthetic data - RED phase."""
        backend = CPUBackend()

        # Create simple test data
        # Y: 10 voxels, 20 subjects
        # X: 20 subjects, 3 regressors (intercept, age, group)
        n_voxels, n_subjects, n_regressors = 10, 20, 3

        np.random.seed(42)  # For reproducible tests
        Y = np.random.randn(n_voxels, n_subjects)
        X = np.random.randn(n_subjects, n_regressors)
        X[:, 0] = 1  # Intercept column

        # Single contrast (test age effect)
        contrasts = np.array([[0, 1, 0]])  # Test second regressor (age)

        result = backend.compute_glm(Y, X, contrasts)

        # Check result structure
        assert "beta" in result
        assert "t_stat" in result
        assert "p_values" in result
        assert "residuals" in result

        # Check shapes
        assert result["beta"].shape == (n_voxels, n_regressors)
        assert result["t_stat"].shape == (n_voxels, 1)  # One contrast
        assert result["p_values"].shape == (n_voxels, 1)
        assert result["residuals"].shape == (n_voxels, n_subjects)

    def test_compute_glm_with_multiple_contrasts(self):
        """Test GLM computation with multiple contrasts - RED phase."""
        backend = CPUBackend()

        n_voxels, n_subjects, n_regressors = 5, 15, 3

        np.random.seed(123)
        Y = np.random.randn(n_voxels, n_subjects)
        X = np.random.randn(n_subjects, n_regressors)
        X[:, 0] = 1  # Intercept

        # Multiple contrasts
        contrasts = np.array(
            [
                [0, 1, 0],  # Age effect
                [0, 0, 1],  # Group effect
                [0, 1, -1],  # Age vs Group
            ]
        )

        result = backend.compute_glm(Y, X, contrasts)

        # Check shapes for multiple contrasts
        assert result["t_stat"].shape == (n_voxels, 3)  # Three contrasts
        assert result["p_values"].shape == (n_voxels, 3)

    def test_compute_glm_validates_input_dimensions(self):
        """Test that GLM validates input dimensions - RED phase."""
        backend = CPUBackend()

        # Mismatched dimensions
        Y = np.random.randn(10, 20)  # 10 voxels, 20 subjects
        X = np.random.randn(15, 3)  # 15 subjects, 3 regressors (MISMATCH!)
        contrasts = np.array([[0, 1, 0]])

        with pytest.raises(ValueError, match="Dimension mismatch"):
            backend.compute_glm(Y, X, contrasts)

    def test_compute_glm_validates_contrast_dimensions(self):
        """Test that GLM validates contrast dimensions - RED phase."""
        backend = CPUBackend()

        Y = np.random.randn(10, 20)
        X = np.random.randn(20, 3)
        # Wrong number of regressors in contrast
        contrasts = np.array([[0, 1, 0, 1]])  # 4 regressors, but X has 3!

        with pytest.raises(ValueError, match="Contrast dimension mismatch"):
            backend.compute_glm(Y, X, contrasts)

    def test_compute_glm_handles_edge_cases(self):
        """Test GLM handles edge cases - RED phase."""
        backend = CPUBackend()

        # Single voxel, single subject
        Y = np.array([[1.5]])  # 1 voxel, 1 subject
        X = np.array([[1]])  # 1 subject, 1 regressor (intercept only)
        contrasts = np.array([[1]])  # Test intercept

        result = backend.compute_glm(Y, X, contrasts)

        assert result["beta"].shape == (1, 1)
        assert result["t_stat"].shape == (1, 1)


class TestCPUBackendIntegration:
    """Test CPU backend integration with other components - RED phase."""

    def test_cpu_backend_with_real_neuroimaging_dimensions(self):
        """Test CPU backend with realistic neuroimaging data dimensions - RED phase."""
        backend = CPUBackend()

        # Realistic dimensions: 50,000 voxels, 30 subjects, 5 regressors
        n_voxels, n_subjects, n_regressors = 50000, 30, 5

        np.random.seed(42)
        Y = np.random.randn(n_voxels, n_subjects)
        X = np.random.randn(n_subjects, n_regressors)
        X[:, 0] = 1  # Intercept

        contrasts = np.array([[0, 1, 0, 0, 0]])  # Test one regressor

        result = backend.compute_glm(Y, X, contrasts)

        # Should handle large data without issues
        assert result["beta"].shape == (n_voxels, n_regressors)
        assert result["t_stat"].shape == (n_voxels, 1)
        assert np.all(np.isfinite(result["t_stat"]))  # No NaN or Inf values
        assert np.all(np.isfinite(result["p_values"]))

    def test_cpu_backend_numerical_accuracy(self):
        """Test numerical accuracy against known results - RED phase."""
        backend = CPUBackend()

        # Create data with known linear relationship
        np.random.seed(0)
        n_subjects = 100
        X = np.ones((n_subjects, 2))  # Intercept + one regressor
        X[:, 1] = np.linspace(0, 1, n_subjects)  # Linear predictor

        # Create Y with known relationship: Y = 2 + 3*X + noise
        true_beta = np.array([2.0, 3.0])
        Y_clean = X @ true_beta
        noise = 0.1 * np.random.randn(n_subjects)
        Y = (Y_clean + noise).reshape(1, -1)  # Single voxel

        contrasts = np.array([[0, 1]])  # Test slope

        result = backend.compute_glm(Y, X, contrasts)

        # Beta should be close to true values (within noise tolerance)
        estimated_beta = result["beta"][0, :]
        assert np.abs(estimated_beta[0] - 2.0) < 0.5  # Intercept
        assert np.abs(estimated_beta[1] - 3.0) < 0.5  # Slope

        # T-statistic should be significant (slope clearly non-zero)
        assert np.abs(result["t_stat"][0, 0]) > 2.0  # Should be significant

    def test_cpu_backend_memory_efficiency(self):
        """Test that CPU backend doesn't use excessive memory - RED phase."""
        import os

        import psutil

        backend = CPUBackend()

        # Measure initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process moderately large dataset
        n_voxels, n_subjects = 10000, 50
        Y = np.random.randn(n_voxels, n_subjects)
        X = np.random.randn(n_subjects, 3)
        X[:, 0] = 1
        contrasts = np.array([[0, 1, 0]])

        result = backend.compute_glm(Y, X, contrasts)

        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB for this dataset)
        assert memory_increase < 500, f"Memory usage too high: {memory_increase}MB"

        # Clean up
        del result, Y, X
