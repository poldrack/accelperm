"""Tests for contrast file I/O operations - TDD RED phase."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from accelperm.io.contrast import (
    ContrastLoader,
    create_f_contrast,
    create_t_contrast,
    load_contrast_matrix,
    validate_contrast_compatibility,
)


class TestContrastLoader:
    """Test the ContrastLoader class - RED phase."""

    def test_contrast_loader_exists(self):
        """Test that ContrastLoader class exists - RED phase."""
        # This should fail because ContrastLoader doesn't exist yet
        loader = ContrastLoader()
        assert isinstance(loader, ContrastLoader)

    def test_load_fsl_contrast_file(self):
        """Test loading FSL .con contrast files - RED phase."""
        # This should fail because FSL contrast loading doesn't exist yet
        loader = ContrastLoader()

        # Create mock FSL contrast file content (space-separated, no headers)
        fsl_content = "1 -1 0 0\n0 0 1 -1\n1 1 -1 -1"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".con", delete=False) as f:
            f.write(fsl_content)
            contrast_path = Path(f.name)

        try:
            result = loader.load(contrast_path)

            assert "contrast_matrix" in result
            assert "n_contrasts" in result
            assert "n_regressors" in result
            assert "contrast_names" in result

            # Check shape and content
            assert result["contrast_matrix"].shape == (
                3,
                4,
            )  # 3 contrasts, 4 regressors
            assert result["n_contrasts"] == 3
            assert result["n_regressors"] == 4
            assert len(result["contrast_names"]) == 3
        finally:
            contrast_path.unlink()

    def test_load_csv_contrast_file(self):
        """Test loading CSV contrast files with headers - RED phase."""
        # This should fail because CSV contrast loading doesn't exist yet
        loader = ContrastLoader()

        # Create mock CSV contrast file
        csv_content = "name,intercept,age,group_A,group_B\nage_effect,0,1,0,0\ngroup_diff,0,0,1,-1"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            contrast_path = Path(f.name)

        try:
            result = loader.load(contrast_path)

            assert result["contrast_matrix"].shape == (
                2,
                4,
            )  # 2 contrasts, 4 regressors
            assert "age_effect" in result["contrast_names"]
            assert "group_diff" in result["contrast_names"]
        finally:
            contrast_path.unlink()

    def test_load_txt_contrast_file(self):
        """Test loading simple text contrast files - RED phase."""
        # This should fail because text contrast loading doesn't exist yet
        loader = ContrastLoader()

        # Create mock text file (tab-separated)
        txt_content = "1\t0\t0\n0\t1\t0\n0\t0\t1"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(txt_content)
            contrast_path = Path(f.name)

        try:
            result = loader.load(contrast_path)

            assert result["contrast_matrix"].shape == (3, 3)  # 3x3 identity-like matrix
            assert np.allclose(result["contrast_matrix"], np.eye(3))
        finally:
            contrast_path.unlink()

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises appropriate error - RED phase."""
        # This should fail because error handling doesn't exist yet
        loader = ContrastLoader()

        nonexistent_path = Path("/fake/nonexistent.con")

        with pytest.raises(FileNotFoundError, match="Contrast file not found"):
            loader.load(nonexistent_path)

    def test_load_invalid_format_raises_error(self):
        """Test that invalid file format raises appropriate error - RED phase."""
        # This should fail because format validation doesn't exist yet
        loader = ContrastLoader()

        # Create file with unsupported extension
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("1 0 -1")
            invalid_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unsupported contrast file format"):
                loader.load(invalid_path)
        finally:
            invalid_path.unlink()

    def test_validate_contrast_dimensions(self):
        """Test validation of contrast matrix dimensions - RED phase."""
        # This should fail because validation doesn't exist yet
        loader = ContrastLoader(validate_contrasts=True)

        # Create contrast with wrong number of regressors
        wrong_content = "1 0\n0 1 0"  # Inconsistent number of columns

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(wrong_content)
            contrast_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Inconsistent number of regressors"):
                loader.load(contrast_path)
        finally:
            contrast_path.unlink()

    def test_load_with_custom_names(self):
        """Test loading contrasts with custom names - RED phase."""
        # This should fail because custom naming doesn't exist yet
        loader = ContrastLoader()

        contrast_content = "1 0 -1\n0 1 0"
        contrast_names = ["contrast_A", "contrast_B"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(contrast_content)
            contrast_path = Path(f.name)

        try:
            result = loader.load(contrast_path, contrast_names=contrast_names)

            assert result["contrast_names"] == contrast_names
        finally:
            contrast_path.unlink()


class TestContrastUtilityFunctions:
    """Test utility functions for contrast operations - RED phase."""

    def test_load_contrast_matrix_function_exists(self):
        """Test that load_contrast_matrix function exists - RED phase."""
        # This should fail because function doesn't exist yet
        contrast_content = "1 0 -1\n0 1 0"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(contrast_content)
            contrast_path = Path(f.name)

        try:
            contrast_matrix, contrast_names = load_contrast_matrix(contrast_path)

            assert isinstance(contrast_matrix, np.ndarray)
            assert isinstance(contrast_names, list)
            assert contrast_matrix.shape == (2, 3)
            assert len(contrast_names) == 2
        finally:
            contrast_path.unlink()

    def test_validate_contrast_compatibility_function_exists(self):
        """Test contrast-design matrix compatibility validation - RED phase."""
        # This should fail because validation doesn't exist yet

        # Compatible contrast and design
        contrast_matrix = np.array([[1, 0, -1], [0, 1, 0]])
        design_info = {"n_regressors": 3, "column_names": ["intercept", "age", "group"]}

        is_compatible, issues = validate_contrast_compatibility(
            contrast_matrix, design_info
        )
        assert is_compatible is True
        assert len(issues) == 0

        # Incompatible (wrong number of regressors)
        incompatible_design = {
            "n_regressors": 4,
            "column_names": ["intercept", "age", "group_A", "group_B"],
        }

        is_compatible, issues = validate_contrast_compatibility(
            contrast_matrix, incompatible_design
        )
        assert is_compatible is False
        assert len(issues) > 0

    def test_create_t_contrast_function_exists(self):
        """Test t-contrast creation function - RED phase."""
        # This should fail because t-contrast creation doesn't exist yet
        column_names = ["intercept", "age", "group_A", "group_B"]
        contrast_spec = {"age": 1}  # Test age effect

        t_contrast = create_t_contrast(column_names, contrast_spec)

        assert isinstance(t_contrast, np.ndarray)
        assert t_contrast.shape == (4,)
        assert t_contrast[1] == 1  # age coefficient
        assert t_contrast[0] == 0  # intercept not involved
        assert t_contrast[2] == 0  # group_A not involved
        assert t_contrast[3] == 0  # group_B not involved

    def test_create_f_contrast_function_exists(self):
        """Test F-contrast creation function - RED phase."""
        # This should fail because F-contrast creation doesn't exist yet
        column_names = ["intercept", "age", "group_A", "group_B"]
        contrast_specs = [
            {"group_A": 1, "group_B": -1},  # Group difference
            {"age": 1},  # Age effect
        ]

        f_contrast = create_f_contrast(column_names, contrast_specs)

        assert isinstance(f_contrast, np.ndarray)
        assert f_contrast.shape == (2, 4)  # 2 contrasts, 4 regressors


class TestContrastValidation:
    """Test contrast validation and quality checks - RED phase."""

    def test_check_contrast_rank(self):
        """Test detection of rank-deficient contrast matrices - RED phase."""
        # This should fail because rank checking doesn't exist yet
        loader = ContrastLoader(validate_contrasts=True)

        # Create rank-deficient contrasts (linearly dependent rows)
        rank_def_content = "1 0 1\n2 0 2\n0 1 0"  # Row 2 = 2 * Row 1

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(rank_def_content)
            contrast_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Contrast matrix is rank deficient"):
                loader.load(contrast_path, check_rank=True)
        finally:
            contrast_path.unlink()

    def test_check_zero_contrasts(self):
        """Test detection of all-zero contrasts - RED phase."""
        # This should fail because zero contrast checking doesn't exist yet
        loader = ContrastLoader(validate_contrasts=True)

        # Create contrast with zero row
        zero_content = "1 0 -1\n0 0 0\n0 1 0"  # Middle row is all zeros

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(zero_content)
            contrast_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Contrast contains all-zero rows"):
                loader.load(contrast_path)
        finally:
            contrast_path.unlink()

    def test_contrast_design_compatibility_check(self):
        """Test checking compatibility with design matrix - RED phase."""
        # This should fail because compatibility checking doesn't exist yet
        loader = ContrastLoader()

        contrast_content = "1 0 -1"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(contrast_content)
            contrast_path = Path(f.name)

        try:
            design_info = {
                "n_regressors": 3,
                "column_names": ["intercept", "age", "group"],
            }

            result = loader.load(contrast_path, design_info=design_info)

            # Should include compatibility information
            assert "design_compatible" in result
            assert result["design_compatible"] is True
        finally:
            contrast_path.unlink()


class TestContrastFormatSupport:
    """Test different contrast file formats - RED phase."""

    def test_fsl_format_compatibility(self):
        """Test FSL randomise .con file format support - RED phase."""
        # This should fail because FSL format support doesn't exist yet
        loader = ContrastLoader(format_style="fsl")

        # FSL format: space-separated, no headers, specific naming conventions
        fsl_content = "1 0 0\n0 1 0\n0 0 1"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".con", delete=False) as f:
            f.write(fsl_content)
            contrast_path = Path(f.name)

        try:
            result = loader.load(contrast_path)

            # Should generate FSL-style contrast names
            expected_names = ["C1", "C2", "C3"]
            assert result["contrast_names"] == expected_names
            assert result["format_style"] == "fsl"
        finally:
            contrast_path.unlink()

    def test_spm_format_compatibility(self):
        """Test SPM .mat contrast file format support - RED phase."""
        # This should fail because SPM format support doesn't exist yet
        loader = ContrastLoader(format_style="spm")

        # SPM might use different conventions
        contrast_content = "1 -1 0\n0 0 1"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(contrast_content)
            contrast_path = Path(f.name)

        try:
            result = loader.load(contrast_path)

            assert result["format_style"] == "spm"
        finally:
            contrast_path.unlink()

    def test_multiple_contrast_files(self):
        """Test loading multiple contrast files - RED phase."""
        # This should fail because multiple file support doesn't exist yet
        loader = ContrastLoader()

        # Create two contrast files
        contrast1_content = "1 0 -1"
        contrast2_content = "0 1 0"

        with tempfile.TemporaryDirectory() as tmpdir:
            contrast1_path = Path(tmpdir) / "contrast1.txt"
            contrast2_path = Path(tmpdir) / "contrast2.txt"

            contrast1_path.write_text(contrast1_content)
            contrast2_path.write_text(contrast2_content)

            result = loader.load_multiple([contrast1_path, contrast2_path])

            # Should combine contrasts
            assert result["contrast_matrix"].shape == (2, 3)
            assert result["n_contrasts"] == 2


class TestContrastCreation:
    """Test programmatic contrast creation - RED phase."""

    def test_create_simple_contrasts(self):
        """Test creating simple effect contrasts - RED phase."""
        # This should fail because contrast creation doesn't exist yet
        loader = ContrastLoader()

        column_names = ["intercept", "age", "condition_A", "condition_B"]

        # Create various contrast types
        contrasts = loader.create_standard_contrasts(column_names)

        assert "main_effects" in contrasts
        assert "interactions" in contrasts
        assert len(contrasts["main_effects"]) > 0

    def test_create_interaction_contrasts(self):
        """Test creating interaction contrasts - RED phase."""
        # This should fail because interaction contrast creation doesn't exist yet
        column_names = ["intercept", "age", "group", "age_x_group"]

        interaction_contrast = create_t_contrast(column_names, {"age_x_group": 1})

        assert interaction_contrast[3] == 1  # Interaction term
        assert np.sum(np.abs(interaction_contrast[:3])) == 0  # Other terms zero

    def test_create_polynomial_contrasts(self):
        """Test creating polynomial contrasts for ordered factors - RED phase."""
        # This should fail because polynomial contrast creation doesn't exist yet
        loader = ContrastLoader()

        # For ordered conditions (e.g., drug doses: 0mg, 10mg, 20mg, 30mg)
        n_levels = 4
        polynomial_contrasts = loader.create_polynomial_contrasts(n_levels)

        assert "linear" in polynomial_contrasts
        assert "quadratic" in polynomial_contrasts
        assert "cubic" in polynomial_contrasts
