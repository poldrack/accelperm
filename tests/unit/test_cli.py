"""Tests for CLI interface - TDD RED phase."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from typer.testing import CliRunner

from accelperm.cli import app


class TestCLIBasic:
    """Test basic CLI functionality - RED phase."""

    def test_cli_app_exists(self):
        """Test that CLI app exists - RED phase."""
        # This should fail because CLI app doesn't exist yet
        assert app is not None

    def test_cli_version_command(self):
        """Test version command - RED phase."""
        runner = CliRunner()
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "accelperm" in result.stdout.lower()
        assert "version" in result.stdout.lower()

    def test_cli_help_command(self):
        """Test help command - RED phase."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "AccelPerm" in result.stdout
        assert "GPU-accelerated permutation testing" in result.stdout

    def test_cli_glm_command_exists(self):
        """Test that GLM command exists - RED phase."""
        runner = CliRunner()
        result = runner.invoke(app, ["glm", "--help"])

        assert result.exit_code == 0
        assert "glm" in result.stdout.lower()
        assert "General Linear Model" in result.stdout


class TestCLIGLMCommand:
    """Test GLM command functionality - RED phase."""

    def test_glm_command_basic_args(self):
        """Test GLM command with basic arguments - RED phase."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock input files
            input_file = Path(tmpdir) / "input.nii.gz"
            design_file = Path(tmpdir) / "design.txt"
            contrast_file = Path(tmpdir) / "contrasts.con"
            output_dir = Path(tmpdir) / "output"

            # Create minimal files
            input_file.touch()
            design_file.write_text("1 0\n1 1\n")
            contrast_file.write_text("0 1\n")

            with patch("accelperm.cli.run_glm") as mock_run_glm:
                mock_run_glm.return_value = {"status": "success"}

                result = runner.invoke(
                    app,
                    [
                        "glm",
                        "--input",
                        str(input_file),
                        "--design",
                        str(design_file),
                        "--contrasts",
                        str(contrast_file),
                        "--output",
                        str(output_dir),
                    ],
                )

                assert result.exit_code == 0
                mock_run_glm.assert_called_once()

    def test_glm_command_missing_required_args(self):
        """Test GLM command with missing required arguments - RED phase."""
        runner = CliRunner()

        # Missing required arguments should show error
        result = runner.invoke(app, ["glm"])

        assert result.exit_code != 0
        # Check both stdout and stderr for error messages
        output = (result.stdout + result.stderr).lower()
        assert "required" in output or "missing" in output or "option" in output

    def test_glm_command_with_backend_selection(self):
        """Test GLM command with backend selection - RED phase."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.nii.gz"
            design_file = Path(tmpdir) / "design.txt"
            contrast_file = Path(tmpdir) / "contrasts.con"
            output_dir = Path(tmpdir) / "output"

            input_file.touch()
            design_file.write_text("1 0\n1 1\n")
            contrast_file.write_text("0 1\n")

            with patch("accelperm.cli.run_glm") as mock_run_glm:
                mock_run_glm.return_value = {"status": "success"}

                result = runner.invoke(
                    app,
                    [
                        "glm",
                        "--input",
                        str(input_file),
                        "--design",
                        str(design_file),
                        "--contrasts",
                        str(contrast_file),
                        "--output",
                        str(output_dir),
                        "--backend",
                        "cpu",
                    ],
                )

                assert result.exit_code == 0
                # Check that backend was passed to run_glm
                call_args = mock_run_glm.call_args
                assert call_args is not None

    def test_glm_command_with_verbose_output(self):
        """Test GLM command with verbose output - RED phase."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.nii.gz"
            design_file = Path(tmpdir) / "design.txt"
            contrast_file = Path(tmpdir) / "contrasts.con"
            output_dir = Path(tmpdir) / "output"

            input_file.touch()
            design_file.write_text("1 0\n1 1\n")
            contrast_file.write_text("0 1\n")

            with patch("accelperm.cli.run_glm") as mock_run_glm:
                mock_run_glm.return_value = {"status": "success"}

                result = runner.invoke(
                    app,
                    [
                        "glm",
                        "--input",
                        str(input_file),
                        "--design",
                        str(design_file),
                        "--contrasts",
                        str(contrast_file),
                        "--output",
                        str(output_dir),
                        "--verbose",
                    ],
                )

                assert result.exit_code == 0


class TestCLIIntegration:
    """Test CLI integration with core components - RED phase."""

    def test_run_glm_function_exists(self):
        """Test that run_glm function exists - RED phase."""
        from accelperm.cli import run_glm

        # Create mock inputs
        config = {
            "input_file": Path("test.nii.gz"),
            "design_file": Path("design.csv"),  # Use supported format
            "contrast_file": Path("contrasts.con"),
            "output_dir": Path("output"),
            "backend": "cpu",
            "verbose": False,
        }

        # Mock all the underlying components
        with patch("accelperm.io.nifti.NiftiLoader.load") as mock_nifti_load:
            with patch(
                "accelperm.io.design.DesignMatrixLoader.load"
            ) as mock_design_load:
                with patch(
                    "accelperm.io.contrast.ContrastLoader.load"
                ) as mock_contrast_load:
                    with patch(
                        "accelperm.backends.cpu.CPUBackend.compute_glm"
                    ) as mock_compute_glm:
                        with patch(
                            "accelperm.backends.cpu.CPUBackend.is_available"
                        ) as mock_is_available:
                            with patch(
                                "accelperm.io.output.OutputWriter.save_statistical_map"
                            ) as mock_save_stat:
                                with patch(
                                    "accelperm.io.output.OutputWriter.save_p_value_map"
                                ) as mock_save_p:
                                    with patch(
                                        "accelperm.io.output.create_results_summary"
                                    ) as mock_summary:
                                        # Configure mocks with realistic neuroimaging data
                                        mock_nifti_load.return_value = {
                                            "data": np.random.randn(
                                                1000, 20
                                            ),  # 1000 voxels, 20 subjects
                                            "affine": np.eye(4),
                                        }

                                        mock_design_load.return_value = {
                                            "design_matrix": np.random.randn(
                                                20, 3
                                            )  # 20 subjects, 3 regressors
                                        }

                                        mock_contrast_load.return_value = {
                                            "contrast_matrix": np.array(
                                                [[0, 1, 0]]
                                            ),  # 1 contrast, 3 regressors
                                            "contrast_names": ["test_contrast"],
                                        }

                                        mock_compute_glm.return_value = {
                                            "beta": np.random.randn(
                                                1000, 3
                                            ),  # 1000 voxels, 3 regressors
                                            "t_stat": np.random.randn(
                                                1000, 1
                                            ),  # 1000 voxels, 1 contrast
                                            "p_values": np.random.rand(
                                                1000, 1
                                            ),  # 1000 voxels, 1 contrast
                                        }
                                        mock_is_available.return_value = True

                                        result = run_glm(config)

                                        assert "status" in result
                                        if result["status"] != "success":
                                            print(
                                                f"Error: {result.get('error', 'No error message')}"
                                            )
                                        assert result["status"] == "success"

    def test_cli_error_handling(self):
        """Test CLI error handling - RED phase."""
        runner = CliRunner()

        # Test with non-existent input file
        result = runner.invoke(
            app,
            [
                "glm",
                "--input",
                "/nonexistent/file.nii.gz",
                "--design",
                "/nonexistent/design.txt",
                "--contrasts",
                "/nonexistent/contrasts.con",
                "--output",
                "/tmp/output",
            ],
        )

        # Should handle error gracefully
        assert result.exit_code != 0

    def test_cli_backend_validation(self):
        """Test CLI validates backend selection - RED phase."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.nii.gz"
            design_file = Path(tmpdir) / "design.txt"
            contrast_file = Path(tmpdir) / "contrasts.con"
            output_dir = Path(tmpdir) / "output"

            input_file.touch()
            design_file.write_text("1 0\n1 1\n")
            contrast_file.write_text("0 1\n")

            # Test with invalid backend
            result = runner.invoke(
                app,
                [
                    "glm",
                    "--input",
                    str(input_file),
                    "--design",
                    str(design_file),
                    "--contrasts",
                    str(contrast_file),
                    "--output",
                    str(output_dir),
                    "--backend",
                    "invalid_backend",
                ],
            )

            # Should show error for invalid backend
            assert result.exit_code != 0
