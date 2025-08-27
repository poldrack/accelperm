"""Simplified tests for output writer I/O operations - TDD approach."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from accelperm.io.output import (
    OutputWriter,
    create_results_summary,
    generate_cluster_table,
    save_p_value_map,
    save_statistical_map,
)


class TestOutputWriterCore:
    """Test core OutputWriter functionality."""

    def test_output_writer_exists(self):
        """Test that OutputWriter class exists."""
        writer = OutputWriter()
        assert isinstance(writer, OutputWriter)
        assert writer.format_style == "standard"
        assert writer.include_metadata is False
        assert writer.validate_output is False

    def test_output_writer_with_options(self):
        """Test OutputWriter with different options."""
        writer = OutputWriter(
            format_style="fsl", include_metadata=True, validate_output=True
        )
        assert writer.format_style == "fsl"
        assert writer.include_metadata is True
        assert writer.validate_output is True

    def test_save_statistical_map_calls_nifti_save(self):
        """Test that save_statistical_map calls the NIfTI save function."""
        writer = OutputWriter()
        stat_map = np.random.randn(10, 10, 10)
        affine = np.eye(4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.nii.gz"

            with patch("accelperm.io.output.save_nifti") as mock_save:
                writer.save_statistical_map(stat_map, affine, output_path, "t-stat")
                mock_save.assert_called_once_with(stat_map, affine, output_path)

    def test_save_p_value_map_calls_nifti_save(self):
        """Test that save_p_value_map calls the NIfTI save function."""
        writer = OutputWriter()
        p_values = np.random.rand(10, 10, 10)
        affine = np.eye(4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pvals.nii.gz"

            with patch("accelperm.io.output.save_nifti") as mock_save:
                writer.save_p_value_map(p_values, affine, output_path)
                mock_save.assert_called_once_with(p_values, affine, output_path)

    def test_generate_cluster_table_creates_file(self):
        """Test cluster table generation."""
        writer = OutputWriter()
        clusters = [
            {
                "size": 100,
                "peak_coord": (10, 20, 30),
                "peak_stat": 4.5,
                "p_corrected": 0.001,
            },
            {
                "size": 50,
                "peak_coord": (-10, 15, 25),
                "peak_stat": 3.2,
                "p_corrected": 0.01,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "clusters.txt"
            writer.generate_cluster_table(clusters, output_path)

            # Check file was created and has content
            assert output_path.exists()
            content = output_path.read_text()
            assert "Cluster Table" in content
            assert "100" in content  # First cluster size
            assert "4.500" in content  # First cluster stat


class TestOutputValidation:
    """Test output validation functionality."""

    def test_validate_statistical_map_with_nan(self):
        """Test validation catches NaN values in statistical maps."""
        writer = OutputWriter(validate_output=True)

        # Create invalid map with NaN
        stat_map = np.ones((5, 5, 5))
        stat_map[2, 2, 2] = np.nan
        affine = np.eye(4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "invalid.nii.gz"

            with pytest.raises(
                ValueError, match="Statistical map contains invalid values"
            ):
                writer.save_statistical_map(stat_map, affine, output_path, "t-stat")

    def test_validate_p_value_range(self):
        """Test validation catches invalid p-value ranges."""
        writer = OutputWriter(validate_output=True)

        # Create invalid p-values (>1)
        p_values = np.ones((5, 5, 5)) * 0.5
        p_values[2, 2, 2] = 1.5  # Invalid
        affine = np.eye(4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "invalid_pvals.nii.gz"

            with pytest.raises(ValueError, match="P-values must be between 0 and 1"):
                writer.save_p_value_map(p_values, affine, output_path)

    def test_check_output_completeness(self):
        """Test checking for complete output files."""
        writer = OutputWriter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create some files
            (output_dir / "tstat1.nii.gz").touch()
            (output_dir / "clusters.txt").touch()

            # Check completeness
            expected_files = ["tstat1.nii.gz", "clusters.txt"]
            assert writer.check_output_completeness(output_dir, expected_files) is True

            # Check with missing file
            expected_files.append("missing.nii.gz")
            assert writer.check_output_completeness(output_dir, expected_files) is False


class TestUtilityFunctions:
    """Test standalone utility functions."""

    def test_save_statistical_map_function(self):
        """Test standalone save_statistical_map function."""
        stat_map = np.random.randn(5, 5, 5)
        affine = np.eye(4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_stat.nii.gz"

            with patch("accelperm.io.output.save_nifti") as mock_save:
                save_statistical_map(stat_map, affine, output_path)
                mock_save.assert_called_once_with(stat_map, affine, output_path)

    def test_save_p_value_map_function(self):
        """Test standalone save_p_value_map function."""
        p_values = np.random.rand(5, 5, 5)
        affine = np.eye(4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_pvals.nii.gz"

            with patch("accelperm.io.output.save_nifti") as mock_save:
                save_p_value_map(p_values, affine, output_path)
                mock_save.assert_called_once_with(p_values, affine, output_path)

    def test_generate_cluster_table_function(self):
        """Test standalone generate_cluster_table function."""
        clusters = [
            {
                "size": 75,
                "peak_coord": (5, 10, 15),
                "peak_stat": 3.5,
                "p_corrected": 0.005,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_clusters.txt"

            generate_cluster_table(clusters, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "75" in content

    def test_create_results_summary_function(self):
        """Test standalone create_results_summary function."""
        results = {
            "n_permutations": 5000,
            "significant_voxels": 1250,
            "max_statistic": 6.78,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "summary.txt"

            create_results_summary(results, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "n_permutations: 5000" in content
            assert "AccelPerm" in content


class TestBatchOperations:
    """Test batch output operations."""

    def test_save_multiple_maps(self):
        """Test saving multiple maps at once."""
        writer = OutputWriter()

        maps = {
            "tstat1": np.random.randn(5, 5, 5),
            "tstat2": np.random.randn(5, 5, 5),
        }
        affine = np.eye(4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch("accelperm.io.output.save_nifti") as mock_save:
                writer.save_multiple_maps(maps, affine, output_dir)

                # Should be called once for each map
                assert mock_save.call_count == 2

    def test_create_analysis_log(self):
        """Test creating analysis log in JSON format."""
        writer = OutputWriter()

        log_data = {
            "start_time": "2025-08-27T10:00:00",
            "n_permutations": 1000,
            "results": {"max_stat": 5.5},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "log.json"

            writer.create_analysis_log(log_data, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "n_permutations" in content
            assert "1000" in content
