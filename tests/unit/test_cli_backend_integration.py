"""Tests for CLI backend factory integration - TDD RED phase."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from accelperm.cli import run_glm


class TestCLIBackendIntegration:
    """Test CLI integration with backend factory - RED phase."""

    def test_cli_uses_backend_factory(self):
        """Test CLI uses backend factory for backend selection - RED phase."""
        # Create temporary test files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock NIfTI file
            input_file = tmpdir / "test_data.nii.gz"
            input_file.touch()

            # Create mock design file
            design_file = tmpdir / "design.txt"
            design_file.write_text("1.0 0.0\n1.0 1.0\n")

            # Create mock contrast file
            contrast_file = tmpdir / "contrasts.con"
            contrast_file.write_text("/NumWaves 2\n/NumContrasts 1\n0.0 1.0\n")

            # Create output directory
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            config = {
                "input_file": input_file,
                "design_file": design_file,
                "contrast_file": contrast_file,
                "output_dir": output_dir,
                "backend": "auto",
                "verbose": False,
            }

            # Mock the data loading and computation
            with patch("accelperm.io.nifti.NiftiLoader") as mock_nifti, patch(
                "accelperm.io.design.DesignMatrixLoader"
            ) as mock_design, patch(
                "accelperm.io.contrast.ContrastLoader"
            ) as mock_contrast, patch(
                "accelperm.io.output.OutputWriter"
            ) as mock_output:
                # Mock data loading
                mock_nifti_instance = mock_nifti.return_value
                mock_nifti_instance.load.return_value = {
                    "data": np.random.randn(100, 2),
                    "affine": np.eye(4),
                }

                mock_design_instance = mock_design.return_value
                mock_design_instance.load.return_value = {
                    "design_matrix": np.array([[1.0, 0.0], [1.0, 1.0]])
                }

                mock_contrast_instance = mock_contrast.return_value
                mock_contrast_instance.load.return_value = {
                    "contrast_matrix": np.array([[0.0, 1.0]]),
                    "contrast_names": ["test_contrast"],
                }

                mock_output_instance = mock_output.return_value
                mock_output_instance.save_statistical_map = MagicMock()
                mock_output_instance.save_p_value_map = MagicMock()

                # Run GLM
                result = run_glm(config)

                # Should complete successfully
                assert result["status"] == "success"
                assert "results" in result

    def test_cli_respects_backend_preference(self):
        """Test CLI respects user backend preference - RED phase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create minimal test files
            input_file = tmpdir / "test_data.nii.gz"
            input_file.touch()
            design_file = tmpdir / "design.txt"
            design_file.write_text("1.0 0.0\n1.0 1.0\n")
            contrast_file = tmpdir / "contrasts.con"
            contrast_file.write_text("/NumWaves 2\n/NumContrasts 1\n0.0 1.0\n")
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            config_cpu = {
                "input_file": input_file,
                "design_file": design_file,
                "contrast_file": contrast_file,
                "output_dir": output_dir,
                "backend": "cpu",
                "verbose": False,
            }

            # Mock all the I/O operations
            with patch("accelperm.io.nifti.NiftiLoader") as mock_nifti, patch(
                "accelperm.io.design.DesignMatrixLoader"
            ) as mock_design, patch(
                "accelperm.io.contrast.ContrastLoader"
            ) as mock_contrast, patch("accelperm.io.output.OutputWriter"):
                # Setup mocks
                mock_nifti.return_value.load.return_value = {
                    "data": np.random.randn(100, 2),
                    "affine": np.eye(4),
                }
                mock_design.return_value.load.return_value = {
                    "design_matrix": np.array([[1.0, 0.0], [1.0, 1.0]])
                }
                mock_contrast.return_value.load.return_value = {
                    "contrast_matrix": np.array([[0.0, 1.0]]),
                    "contrast_names": ["test"],
                }

                # Mock BackendFactory to track which backend was requested
                with patch("accelperm.cli.BackendFactory") as mock_factory:
                    mock_factory_instance = mock_factory.return_value
                    mock_backend = MagicMock()
                    mock_backend.name = "cpu"
                    mock_backend.compute_glm.return_value = {
                        "beta": np.random.randn(100, 2),
                        "t_stat": np.random.randn(100, 1),
                        "p_values": np.random.randn(100, 1),
                    }

                    mock_factory_instance.get_backend.return_value = mock_backend
                    mock_factory_instance.get_backend_capabilities.return_value = {
                        "max_memory_gb": 16.0
                    }

                    result = run_glm(config_cpu)

                    # Should request CPU backend specifically
                    mock_factory_instance.get_backend.assert_called_with("cpu")
                    assert result["status"] == "success"

    def test_cli_auto_backend_selection(self):
        """Test CLI auto backend selection - RED phase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create minimal test files
            input_file = tmpdir / "test_data.nii.gz"
            input_file.touch()
            design_file = tmpdir / "design.txt"
            design_file.write_text("1.0 0.0\n1.0 1.0\n")
            contrast_file = tmpdir / "contrasts.con"
            contrast_file.write_text("/NumWaves 2\n/NumContrasts 1\n0.0 1.0\n")
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            config_auto = {
                "input_file": input_file,
                "design_file": design_file,
                "contrast_file": contrast_file,
                "output_dir": output_dir,
                "backend": "auto",
                "verbose": False,
            }

            # Mock all operations
            with patch("accelperm.io.nifti.NiftiLoader") as mock_nifti, patch(
                "accelperm.io.design.DesignMatrixLoader"
            ) as mock_design, patch(
                "accelperm.io.contrast.ContrastLoader"
            ) as mock_contrast, patch("accelperm.io.output.OutputWriter"):
                # Setup mocks
                mock_nifti.return_value.load.return_value = {
                    "data": np.random.randn(100, 2),
                    "affine": np.eye(4),
                }
                mock_design.return_value.load.return_value = {
                    "design_matrix": np.array([[1.0, 0.0], [1.0, 1.0]])
                }
                mock_contrast.return_value.load.return_value = {
                    "contrast_matrix": np.array([[0.0, 1.0]]),
                    "contrast_names": ["test"],
                }

                # Mock BackendFactory for auto selection
                with patch("accelperm.cli.BackendFactory") as mock_factory:
                    mock_factory_instance = mock_factory.return_value
                    mock_backend = MagicMock()
                    mock_backend.name = "mps"  # Simulate MPS being selected
                    mock_backend.compute_glm.return_value = {
                        "beta": np.random.randn(100, 2),
                        "t_stat": np.random.randn(100, 1),
                        "p_values": np.random.randn(100, 1),
                    }

                    mock_factory_instance.get_best_backend.return_value = mock_backend
                    mock_factory_instance.get_backend_capabilities.return_value = {
                        "max_memory_gb": 64.0
                    }

                    result = run_glm(config_auto)

                    # Should use auto selection
                    mock_factory_instance.get_best_backend.assert_called_once()
                    assert result["status"] == "success"
