"""Configuration management using Pydantic with TOML support for AccelPerm."""

import os
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any

import tomli_w
import torch
from pydantic import BaseModel, Field, field_validator

from accelperm.utils.logging import AccelPermLogger, LogLevel


class ValidationLevel(Enum):
    """Validation strictness levels."""

    STRICT = "strict"
    NORMAL = "normal"
    PERMISSIVE = "permissive"


class BackendConfig(BaseModel):
    """Configuration for compute backend selection."""

    preferred_backend: str = Field(
        default="auto", description="Preferred compute backend"
    )
    fallback_to_cpu: bool = Field(
        default=True, description="Fallback to CPU if preferred unavailable"
    )
    memory_limit_gb: float = Field(default=8.0, description="Memory limit in GB")

    @field_validator("preferred_backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Validate that backend is supported."""
        valid_backends = {"auto", "cpu", "cuda", "mps"}
        if v not in valid_backends:
            raise ValueError(f"Invalid backend '{v}'. Must be one of {valid_backends}")
        return v

    @field_validator("memory_limit_gb")
    @classmethod
    def validate_memory_limit(cls, v: float) -> float:
        """Validate that memory limit is positive."""
        if v <= 0:
            raise ValueError("memory_limit_gb must be positive")
        return v


class ComputeConfig(BaseModel):
    """Configuration for computation parameters."""

    n_permutations: int = Field(default=1000, description="Number of permutations")
    chunk_size: int = Field(default=512, description="Chunk size for processing")
    n_threads: int = Field(
        default=4, description="Number of threads for CPU operations"
    )

    @field_validator("n_permutations")
    @classmethod
    def validate_n_permutations(cls, v: int) -> int:
        """Validate that n_permutations is positive."""
        if v <= 0:
            raise ValueError("n_permutations must be positive")
        return v

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        """Validate that chunk_size is positive."""
        if v <= 0:
            raise ValueError("chunk_size must be positive")
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging behavior."""

    level: str = Field(default="INFO", description="Log level")
    colored_output: bool = Field(
        default=True, description="Enable colored console output"
    )
    progress_reporting: bool = Field(
        default=True, description="Enable progress reporting"
    )

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate that log level is supported."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR"}
        if v not in valid_levels:
            raise ValueError(f"Invalid log level '{v}'. Must be one of {valid_levels}")
        return v


class OutputConfig(BaseModel):
    """Configuration for output handling."""

    base_directory: Path = Field(
        default=Path("./output"), description="Base output directory"
    )
    prefix: str = Field(default="accelperm", description="Prefix for output files")
    save_intermediate: bool = Field(
        default=False, description="Save intermediate results"
    )

    @field_validator("base_directory", mode="before")
    @classmethod
    def validate_base_directory(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class AccelPermConfig(BaseModel):
    """Main configuration model for AccelPerm."""

    backend: BackendConfig = Field(default_factory=BackendConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    validation_level: ValidationLevel = Field(default=ValidationLevel.NORMAL)

    def create_logger(self, name: str) -> AccelPermLogger:
        """Create a logger based on configuration settings."""
        # Convert string level to LogLevel enum
        level_mapping = {
            "DEBUG": LogLevel.DEBUG,
            "INFO": LogLevel.INFO,
            "WARNING": LogLevel.WARNING,
            "ERROR": LogLevel.ERROR,
        }
        log_level = level_mapping[self.logging.level]

        return AccelPermLogger(
            name=name, level=log_level, colored=self.logging.colored_output
        )

    def validate_backend_availability(self) -> bool:
        """Check if the preferred backend is available."""
        if self.backend.preferred_backend == "auto":
            return True  # Auto-selection always works
        elif self.backend.preferred_backend == "cpu":
            return True  # CPU always available
        elif self.backend.preferred_backend == "cuda":
            return torch.cuda.is_available()
        elif self.backend.preferred_backend == "mps":
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return False


def _load_from_env() -> dict[str, dict[str, Any]]:
    """Load configuration values from environment variables."""
    config_dict: dict[str, dict[str, Any]] = {}

    # Define environment variable mappings
    env_mappings = {
        "ACCELPERM_BACKEND_PREFERRED_BACKEND": ("backend", "preferred_backend"),
        "ACCELPERM_BACKEND_MEMORY_LIMIT_GB": ("backend", "memory_limit_gb", float),
        "ACCELPERM_BACKEND_FALLBACK_TO_CPU": ("backend", "fallback_to_cpu", bool),
        "ACCELPERM_COMPUTE_N_PERMUTATIONS": ("compute", "n_permutations", int),
        "ACCELPERM_COMPUTE_CHUNK_SIZE": ("compute", "chunk_size", int),
        "ACCELPERM_COMPUTE_N_THREADS": ("compute", "n_threads", int),
        "ACCELPERM_LOGGING_LEVEL": ("logging", "level"),
        "ACCELPERM_LOGGING_COLORED_OUTPUT": ("logging", "colored_output", bool),
        "ACCELPERM_LOGGING_PROGRESS_REPORTING": ("logging", "progress_reporting", bool),
        "ACCELPERM_OUTPUT_BASE_DIRECTORY": ("output", "base_directory"),
        "ACCELPERM_OUTPUT_PREFIX": ("output", "prefix"),
        "ACCELPERM_OUTPUT_SAVE_INTERMEDIATE": ("output", "save_intermediate", bool),
    }

    for env_var, mapping in env_mappings.items():
        if env_var in os.environ:
            str_value = os.environ[env_var]

            # Convert value if type converter specified
            converted_value: Any = str_value
            if len(mapping) > 2:
                converter = mapping[2]
                if converter == bool:
                    converted_value = str_value.lower() in ("true", "1", "yes", "on")
                elif converter == int:
                    converted_value = int(str_value)
                elif converter == float:
                    converted_value = float(str_value)
                else:
                    converted_value = str_value

            # Set nested config value
            section = str(mapping[0])
            key = str(mapping[1])
            if section not in config_dict:
                config_dict[section] = {}
            config_dict[section][key] = converted_value

    return config_dict


def load_config(config_path: str | None = None) -> AccelPermConfig:
    """Load configuration from TOML file and environment variables."""
    config_dict = {}

    # Load from file if provided and exists
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "rb") as f:
                config_dict = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Invalid TOML file '{config_path}': {e}") from e
    elif config_path and not os.path.exists(config_path):
        # File specified but doesn't exist - return default config
        pass

    # Load from environment variables (overrides file values)
    env_config = _load_from_env()
    for section, values in env_config.items():
        if section not in config_dict:
            config_dict[section] = {}
        config_dict[section].update(values)

    # Create and validate config
    try:
        return AccelPermConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e


def save_config(config: AccelPermConfig, config_path: str) -> None:
    """Save configuration to TOML file."""
    # Convert config to dict and handle Path objects
    config_dict = config.model_dump()

    # Convert Path objects to strings for TOML serialization
    if "output" in config_dict and "base_directory" in config_dict["output"]:
        config_dict["output"]["base_directory"] = str(
            config_dict["output"]["base_directory"]
        )

    # Convert enum to string
    if "validation_level" in config_dict:
        config_dict["validation_level"] = config_dict["validation_level"].value

    with open(config_path, "wb") as f:
        tomli_w.dump(config_dict, f)
