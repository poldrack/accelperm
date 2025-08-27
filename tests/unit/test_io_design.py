"""Tests for design matrix I/O operations - TDD RED phase."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from accelperm.io.design import (
    DesignMatrixLoader,
    load_design_matrix,
    validate_design_matrix,
    create_contrast_matrix,
)


class TestDesignMatrixLoader:
    """Test the DesignMatrixLoader class - RED phase."""

    def test_design_matrix_loader_exists(self):
        """Test that DesignMatrixLoader class exists - RED phase."""
        # This should fail because DesignMatrixLoader doesn't exist yet
        loader = DesignMatrixLoader()
        assert isinstance(loader, DesignMatrixLoader)

    def test_load_csv_design_matrix(self):
        """Test loading design matrix from CSV file - RED phase."""
        # This should fail because CSV loading doesn't exist yet
        loader = DesignMatrixLoader()
        
        # Create mock CSV content with only numeric columns
        csv_content = "subject,age,score\n1,25,85.5\n2,30,92.0\n3,28,78.5"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)
        
        try:
            result = loader.load(csv_path)
            
            assert "design_matrix" in result
            assert "column_names" in result
            assert "n_subjects" in result
            assert "n_regressors" in result
            
            # Check shape and content
            assert result["design_matrix"].shape[0] == 3  # 3 subjects
            assert result["n_subjects"] == 3
            assert "age" in result["column_names"]
            assert "score" in result["column_names"]
        finally:
            csv_path.unlink()

    def test_load_tsv_design_matrix(self):
        """Test loading design matrix from TSV file - RED phase."""
        # This should fail because TSV loading doesn't exist yet
        loader = DesignMatrixLoader()
        
        # Create mock TSV content (tab-separated, numeric only)
        tsv_content = "subject\tage\tscore\n1\t25\t85.5\n2\t30\t92.0\n3\t28\t78.5"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write(tsv_content)
            tsv_path = Path(f.name)
        
        try:
            result = loader.load(tsv_path)
            
            assert "design_matrix" in result
            assert "column_names" in result
            assert result["design_matrix"].shape[0] == 3
            assert result["n_subjects"] == 3
        finally:
            tsv_path.unlink()

    def test_load_design_matrix_with_categorical_encoding(self):
        """Test loading design matrix with automatic categorical encoding - RED phase."""
        # This should fail because categorical encoding doesn't exist yet
        loader = DesignMatrixLoader(encode_categorical=True)
        
        csv_content = "subject,age,group,condition\n1,25,A,control\n2,30,B,treatment\n3,28,A,control"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)
        
        try:
            result = loader.load(csv_path)
            
            # Should have dummy-coded categorical variables
            assert "design_matrix" in result
            assert "categorical_columns" in result
            
            # Check that categorical columns were encoded
            design_matrix = result["design_matrix"]
            assert design_matrix.shape[1] > 3  # More columns due to encoding
        finally:
            csv_path.unlink()

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises appropriate error - RED phase."""
        # This should fail because error handling doesn't exist yet
        loader = DesignMatrixLoader()
        
        nonexistent_path = Path("/fake/nonexistent.csv")
        
        with pytest.raises(FileNotFoundError, match="Design matrix file not found"):
            loader.load(nonexistent_path)

    def test_load_invalid_format_raises_error(self):
        """Test that invalid file format raises appropriate error - RED phase."""
        # This should fail because format validation doesn't exist yet
        loader = DesignMatrixLoader()
        
        # Create file with invalid extension
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some content")
            invalid_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                loader.load(invalid_path)
        finally:
            invalid_path.unlink()

    def test_add_intercept_column(self):
        """Test adding intercept column to design matrix - RED phase."""
        # This should fail because intercept functionality doesn't exist yet
        loader = DesignMatrixLoader(add_intercept=True)
        
        csv_content = "age,group\n25,1\n30,0\n28,1"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)
        
        try:
            result = loader.load(csv_path)
            
            # Should have added intercept column (all ones)
            design_matrix = result["design_matrix"]
            intercept_col = design_matrix[:, 0]  # First column should be intercept
            assert np.all(intercept_col == 1.0)
            assert "intercept" in result["column_names"]
        finally:
            csv_path.unlink()


class TestDesignMatrixUtilityFunctions:
    """Test utility functions for design matrix operations - RED phase."""

    def test_load_design_matrix_function_exists(self):
        """Test that load_design_matrix function exists - RED phase."""
        # This should fail because function doesn't exist yet
        csv_content = "age,group\n25,1\n30,0\n28,1"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)
        
        try:
            design_matrix, column_names = load_design_matrix(csv_path)
            
            assert isinstance(design_matrix, np.ndarray)
            assert isinstance(column_names, list)
            assert design_matrix.shape[0] == 3  # 3 subjects
            assert len(column_names) == design_matrix.shape[1]
        finally:
            csv_path.unlink()

    def test_validate_design_matrix_function_exists(self):
        """Test design matrix validation function - RED phase."""
        # This should fail because validation doesn't exist yet
        
        # Valid design matrix
        valid_matrix = np.array([[1, 25, 1], [1, 30, 0], [1, 28, 1]], dtype=float)
        
        is_valid, issues = validate_design_matrix(valid_matrix)
        assert is_valid is True
        assert len(issues) == 0
        
        # Invalid design matrix (contains NaN)
        invalid_matrix = np.array([[1, 25, 1], [1, np.nan, 0], [1, 28, 1]], dtype=float)
        
        is_valid, issues = validate_design_matrix(invalid_matrix)
        assert is_valid is False
        assert len(issues) > 0
        assert any("NaN" in issue for issue in issues)

    def test_create_contrast_matrix_function_exists(self):
        """Test contrast matrix creation function - RED phase."""
        # This should fail because contrast creation doesn't exist yet
        column_names = ["intercept", "age", "group_A", "group_B"]
        
        # Simple contrast: group_A vs group_B
        contrast = create_contrast_matrix(column_names, {"group_A": 1, "group_B": -1})
        
        assert isinstance(contrast, np.ndarray)
        assert contrast.shape == (len(column_names),)
        assert contrast[2] == 1   # group_A coefficient
        assert contrast[3] == -1  # group_B coefficient
        assert contrast[0] == 0   # intercept not involved
        assert contrast[1] == 0   # age not involved


class TestDesignMatrixValidation:
    """Test design matrix validation and quality checks - RED phase."""

    def test_check_rank_deficiency(self):
        """Test detection of rank-deficient design matrices - RED phase."""
        # This should fail because rank checking doesn't exist yet
        loader = DesignMatrixLoader()
        
        # Create rank-deficient matrix (linearly dependent columns)
        csv_content = "col1,col2,col3\n1,2,3\n2,4,6\n3,6,9"  # col3 = 3*col1
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Design matrix is rank deficient"):
                loader.load(csv_path, check_rank=True)
        finally:
            csv_path.unlink()

    def test_check_missing_values(self):
        """Test detection of missing values in design matrix - RED phase."""
        # This should fail because missing value checking doesn't exist yet
        loader = DesignMatrixLoader()
        
        # Create matrix with missing values (represented as empty strings)
        csv_content = "age,group\n25,A\n,B\n28,A"  # Missing age for subject 2
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Design matrix contains missing values"):
                loader.load(csv_path)
        finally:
            csv_path.unlink()

    def test_check_constant_columns(self):
        """Test detection of constant columns - RED phase."""
        # This should fail because constant column checking doesn't exist yet
        loader = DesignMatrixLoader()
        
        # Create matrix with constant column (only numeric columns)
        csv_content = "age,constant,score\n25,1,85\n30,1,92\n28,1,78"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)
        
        try:
            result = loader.load(csv_path, warn_constant=True)
            assert "warnings" in result
            assert any("constant" in warning.lower() for warning in result["warnings"])
        finally:
            csv_path.unlink()


class TestDesignMatrixCompatibility:
    """Test design matrix compatibility with neuroimaging standards - RED phase."""

    def test_fsl_format_compatibility(self):
        """Test loading FSL-compatible design matrices - RED phase."""
        # This should fail because FSL compatibility doesn't exist yet
        loader = DesignMatrixLoader(format_style="fsl")
        
        # FSL format: no headers, space-separated
        fsl_content = "1 25 1\n1 30 0\n1 28 1"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mat', delete=False) as f:
            f.write(fsl_content)
            fsl_path = Path(f.name)
        
        try:
            result = loader.load(fsl_path)
            
            assert "design_matrix" in result
            assert result["design_matrix"].shape == (3, 3)
            assert result["column_names"] == ["EV1", "EV2", "EV3"]  # FSL naming
        finally:
            fsl_path.unlink()

    def test_spm_format_compatibility(self):
        """Test loading SPM-compatible design matrices - RED phase."""
        # This should fail because SPM compatibility doesn't exist yet
        loader = DesignMatrixLoader(format_style="spm")
        
        # SPM format might have different conventions
        csv_content = "intercept,age,condition\n1,25,1\n1,30,-1\n1,28,1"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            spm_path = Path(f.name)
        
        try:
            result = loader.load(spm_path)
            
            assert "design_matrix" in result
            assert result["format_style"] == "spm"
        finally:
            spm_path.unlink()


class TestDesignMatrixTransformations:
    """Test design matrix transformations and preprocessing - RED phase."""

    def test_standardize_regressors(self):
        """Test standardization of continuous regressors - RED phase."""
        # This should fail because standardization doesn't exist yet
        loader = DesignMatrixLoader(standardize=True)
        
        csv_content = "age,income,group\n25,50000,1\n30,75000,0\n28,60000,1"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)
        
        try:
            result = loader.load(csv_path)
            
            design_matrix = result["design_matrix"]
            
            # Continuous columns should be standardized (mean=0, std=1)
            age_col = design_matrix[:, result["column_names"].index("age")]
            income_col = design_matrix[:, result["column_names"].index("income")]
            
            assert abs(np.mean(age_col)) < 1e-10  # Mean should be ~0
            assert abs(np.std(age_col) - 1.0) < 1e-10  # Std should be ~1
            assert abs(np.mean(income_col)) < 1e-10
            assert abs(np.std(income_col) - 1.0) < 1e-10
        finally:
            csv_path.unlink()

    def test_orthogonalize_regressors(self):
        """Test Gram-Schmidt orthogonalization of regressors - RED phase."""
        # This should fail because orthogonalization doesn't exist yet
        loader = DesignMatrixLoader(orthogonalize=True)
        
        # Create correlated regressors
        csv_content = "reg1,reg2,reg3\n1,2,1.5\n2,4,3\n3,6,4.5\n4,8,6"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)
        
        try:
            result = loader.load(csv_path)
            
            design_matrix = result["design_matrix"]
            
            # Check orthogonality: X.T @ X should be close to diagonal
            gram_matrix = design_matrix.T @ design_matrix
            
            # Off-diagonal elements should be close to zero
            n_cols = gram_matrix.shape[0]
            for i in range(n_cols):
                for j in range(n_cols):
                    if i != j:
                        assert abs(gram_matrix[i, j]) < 1e-10
        finally:
            csv_path.unlink()