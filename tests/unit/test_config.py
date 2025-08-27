"""Tests for configuration management - TDD RED phase."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from accelperm.config import (
    AccelPermConfig,
    BackendConfig,
    ComputeConfig,
    LoggingConfig,
    OutputConfig,
    ValidationLevel,
    load_config,
    save_config,
)


class TestConfigDataModels:
    """Test the Pydantic configuration data models - RED phase."""

    def test_backend_config_model_exists(self):
        """Test that BackendConfig model exists - RED phase."""
        # This should fail because BackendConfig doesn't exist yet
        config = BackendConfig(
            preferred_backend="cuda", fallback_to_cpu=True, memory_limit_gb=8.0
        )
        assert config.preferred_backend == "cuda"
        assert config.fallback_to_cpu is True
        assert config.memory_limit_gb == 8.0

    def test_compute_config_model_exists(self):
        """Test that ComputeConfig model exists - RED phase."""
        # This should fail because ComputeConfig doesn't exist yet
        config = ComputeConfig(n_permutations=1000, chunk_size=512, n_threads=4)
        assert config.n_permutations == 1000
        assert config.chunk_size == 512
        assert config.n_threads == 4

    def test_logging_config_model_exists(self):
        """Test that LoggingConfig model exists - RED phase."""
        # This should fail because LoggingConfig doesn't exist yet
        config = LoggingConfig(
            level="INFO", colored_output=True, progress_reporting=True
        )
        assert config.level == "INFO"
        assert config.colored_output is True
        assert config.progress_reporting is True

    def test_output_config_model_exists(self):
        """Test that OutputConfig model exists - RED phase."""
        # This should fail because OutputConfig doesn't exist yet
        config = OutputConfig(
            base_directory="./results", prefix="accelperm", save_intermediate=False
        )
        assert config.base_directory == Path("./results")
        assert config.prefix == "accelperm"
        assert config.save_intermediate is False

    def test_validation_level_enum_exists(self):
        """Test that ValidationLevel enum exists - RED phase."""
        # This should fail because ValidationLevel doesn't exist yet
        assert hasattr(ValidationLevel, "STRICT")
        assert hasattr(ValidationLevel, "NORMAL")
        assert hasattr(ValidationLevel, "PERMISSIVE")

    def test_main_config_model_exists(self):
        """Test that AccelPermConfig main model exists - RED phase."""
        # This should fail because AccelPermConfig doesn't exist yet
        config = AccelPermConfig(
            backend=BackendConfig(preferred_backend="mps"),
            compute=ComputeConfig(n_permutations=5000),
            logging=LoggingConfig(level="DEBUG"),
            output=OutputConfig(base_directory="/tmp/results"),
            validation_level=ValidationLevel.NORMAL,
        )
        assert config.backend.preferred_backend == "mps"
        assert config.compute.n_permutations == 5000
        assert config.logging.level == "DEBUG"
        assert config.output.base_directory == Path("/tmp/results")
        assert config.validation_level == ValidationLevel.NORMAL


class TestConfigValidation:
    """Test configuration validation and constraints - RED phase."""

    def test_backend_config_validation(self):
        """Test BackendConfig validation rules - RED phase."""
        # This should fail because validation doesn't exist yet

        # Valid backends
        config = BackendConfig(preferred_backend="cuda")
        assert config.preferred_backend == "cuda"

        config = BackendConfig(preferred_backend="mps")
        assert config.preferred_backend == "mps"

        config = BackendConfig(preferred_backend="cpu")
        assert config.preferred_backend == "cpu"

        # Invalid backend should raise ValueError
        with pytest.raises(ValueError, match="Invalid backend"):
            BackendConfig(preferred_backend="invalid_backend")

        # Memory limit validation
        with pytest.raises(ValueError, match="memory_limit_gb must be positive"):
            BackendConfig(memory_limit_gb=-1.0)

    def test_compute_config_validation(self):
        """Test ComputeConfig validation rules - RED phase."""
        # This should fail because validation doesn't exist yet

        # Valid configurations
        config = ComputeConfig(n_permutations=1000)
        assert config.n_permutations == 1000

        # Invalid permutations
        with pytest.raises(ValueError, match="n_permutations must be positive"):
            ComputeConfig(n_permutations=0)

        with pytest.raises(ValueError, match="n_permutations must be positive"):
            ComputeConfig(n_permutations=-100)

        # Invalid chunk size
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ComputeConfig(chunk_size=0)

    def test_logging_config_validation(self):
        """Test LoggingConfig validation rules - RED phase."""
        # This should fail because validation doesn't exist yet

        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            config = LoggingConfig(level=level)
            assert config.level == level

        # Invalid log level
        with pytest.raises(ValueError, match="Invalid log level"):
            LoggingConfig(level="INVALID")

    def test_output_config_path_validation(self):
        """Test OutputConfig path validation - RED phase."""
        # This should fail because validation doesn't exist yet
        config = OutputConfig(base_directory="./test/path")
        assert isinstance(config.base_directory, Path)
        assert str(config.base_directory) == "test/path"


class TestConfigFileOperations:
    """Test configuration file loading and saving - RED phase."""

    def test_load_config_function_exists(self):
        """Test that load_config function exists - RED phase."""
        # This should fail because load_config doesn't exist yet
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[backend]
preferred_backend = "cuda"
fallback_to_cpu = true
memory_limit_gb = 16.0

[compute]
n_permutations = 10000
chunk_size = 1024
n_threads = 8

[logging]
level = "INFO"
colored_output = true
progress_reporting = true

[output]
base_directory = "./output"
prefix = "test"
save_intermediate = false

validation_level = "NORMAL"
"""
            )
            f.flush()

            try:
                config = load_config(f.name)
                assert isinstance(config, AccelPermConfig)
                assert config.backend.preferred_backend == "cuda"
                assert config.compute.n_permutations == 10000
                assert config.logging.level == "INFO"
                assert config.validation_level == ValidationLevel.NORMAL
            finally:
                os.unlink(f.name)

    def test_save_config_function_exists(self):
        """Test that save_config function exists - RED phase."""
        # This should fail because save_config doesn't exist yet
        config = AccelPermConfig(
            backend=BackendConfig(preferred_backend="mps"),
            compute=ComputeConfig(n_permutations=2000),
            logging=LoggingConfig(level="DEBUG"),
            output=OutputConfig(base_directory="/tmp/test"),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            try:
                save_config(config, f.name)

                # Verify file was created and contains expected content
                assert os.path.exists(f.name)

                # Load it back to verify
                loaded_config = load_config(f.name)
                assert loaded_config.backend.preferred_backend == "mps"
                assert loaded_config.compute.n_permutations == 2000
                assert loaded_config.logging.level == "DEBUG"
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_config_from_environment_variables(self):
        """Test loading config from environment variables - RED phase."""
        # This should fail because env var support doesn't exist yet
        env_vars = {
            "ACCELPERM_BACKEND_PREFERRED_BACKEND": "cpu",
            "ACCELPERM_BACKEND_MEMORY_LIMIT_GB": "4.0",
            "ACCELPERM_COMPUTE_N_PERMUTATIONS": "500",
            "ACCELPERM_LOGGING_LEVEL": "WARNING",
            "ACCELPERM_OUTPUT_BASE_DIRECTORY": "/custom/path",
        }

        with patch.dict("os.environ", env_vars):
            config = load_config()  # No file path, should use env vars
            assert config.backend.preferred_backend == "cpu"
            assert config.backend.memory_limit_gb == 4.0
            assert config.compute.n_permutations == 500
            assert config.logging.level == "WARNING"
            assert str(config.output.base_directory) == "/custom/path"

    def test_config_file_not_found_returns_default(self):
        """Test that missing config file returns default config - RED phase."""
        # This should fail because default config handling doesn't exist yet
        config = load_config("nonexistent_file.toml")
        assert isinstance(config, AccelPermConfig)
        # Should have default values
        assert config.backend.preferred_backend == "auto"  # Default
        assert config.compute.n_permutations == 1000  # Default
        assert config.logging.level == "INFO"  # Default

    def test_invalid_toml_file_raises_error(self):
        """Test that invalid TOML file raises appropriate error - RED phase."""
        # This should fail because error handling doesn't exist yet
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("invalid toml content [[[")
            f.flush()

            try:
                with pytest.raises(ValueError, match="Invalid TOML"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)


class TestConfigIntegration:
    """Test config integration with other components - RED phase."""

    def test_config_creates_logger_from_settings(self):
        """Test that config can create logger based on settings - RED phase."""
        # This should fail because integration doesn't exist yet
        config = AccelPermConfig(
            logging=LoggingConfig(
                level="DEBUG", colored_output=False, progress_reporting=True
            )
        )

        logger = config.create_logger("test_logger")
        assert logger.name == "test_logger"
        assert logger.level.value == 10  # DEBUG level
        assert logger.colored is False

    def test_config_validates_backend_availability(self):
        """Test that config validates backend availability - RED phase."""
        # This should fail because backend validation doesn't exist yet
        config = AccelPermConfig(backend=BackendConfig(preferred_backend="cuda"))

        # Should be able to check if backend is available
        with patch("torch.cuda.is_available", return_value=False):
            is_available = config.validate_backend_availability()
            assert is_available is False

        with patch("torch.cuda.is_available", return_value=True):
            is_available = config.validate_backend_availability()
            assert is_available is True
