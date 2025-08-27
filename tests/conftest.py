"""Shared pytest fixtures and configuration."""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_data():
    """Generate sample neuroimaging data for testing."""
    np.random.seed(42)
    n_voxels = 100
    n_subjects = 20
    data = np.random.randn(n_voxels, n_subjects)
    return data


@pytest.fixture
def sample_design_matrix():
    """Generate sample design matrix for testing."""
    np.random.seed(42)
    n_subjects = 20
    n_covariates = 3
    X = np.random.randn(n_subjects, n_covariates)
    # Add intercept column
    X = np.column_stack([np.ones(n_subjects), X])
    return X


@pytest.fixture
def sample_contrast():
    """Generate sample contrast vector for testing."""
    # Test main effect of first covariate (after intercept)
    return np.array([0, 1, 0, 0])


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir