"""Benchmark tests for configuration management."""

import tempfile
from pathlib import Path

from accelperm.config import AccelPermConfig, BackendConfig, load_config, save_config


class TestConfigBenchmarks:
    """Benchmark configuration operations."""

    def test_config_creation_benchmark(self, benchmark):
        """Benchmark configuration object creation."""

        def create_config():
            return AccelPermConfig(
                backend=BackendConfig(preferred_backend="cuda", memory_limit_gb=16.0)
            )

        result = benchmark(create_config)
        assert isinstance(result, AccelPermConfig)

    def test_config_validation_benchmark(self, benchmark):
        """Benchmark configuration validation."""
        config_data = {
            "backend": {
                "preferred_backend": "mps",
                "fallback_to_cpu": True,
                "memory_limit_gb": 8.0,
            },
            "compute": {"n_permutations": 10000, "chunk_size": 1024},
        }

        def validate_config():
            return AccelPermConfig(**config_data)

        result = benchmark(validate_config)
        assert result.backend.preferred_backend == "mps"

    def test_config_file_io_benchmark(self, benchmark):
        """Benchmark configuration file I/O operations."""
        config = AccelPermConfig(backend=BackendConfig(preferred_backend="cpu"))

        def config_round_trip():
            with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
                temp_path = f.name

            try:
                # Save and load config
                save_config(config, temp_path)
                loaded_config = load_config(temp_path)
                return loaded_config
            finally:
                Path(temp_path).unlink(missing_ok=True)

        result = benchmark(config_round_trip)
        assert result.backend.preferred_backend == "cpu"
