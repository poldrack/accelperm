"""Custom exceptions for AccelPerm GPU-accelerated permutation testing."""


class AccelPermError(Exception):
    """Base exception for all AccelPerm errors."""

    pass


class BackendError(AccelPermError):
    """Exception raised for backend-related errors."""

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        """Initialize with optional recovery suggestion.

        Args:
            message: The error message
            suggestion: Optional recovery suggestion
        """
        self.suggestion = suggestion
        full_message = f"{message}\nSuggestion: {suggestion}" if suggestion else message
        super().__init__(full_message)


class DataError(AccelPermError):
    """Exception raised for data-related errors."""

    pass


class ConfigurationError(AccelPermError):
    """Exception raised for configuration-related errors."""

    pass


class InsufficientMemoryError(AccelPermError):
    """Exception raised when there's insufficient memory for operations."""

    def __init__(self, required_gb: float, available_gb: float) -> None:
        """Initialize with memory details.

        Args:
            required_gb: Required memory in GB
            available_gb: Available memory in GB
        """
        self.required_gb = required_gb
        self.available_gb = available_gb
        message = (
            f"Operation requires {required_gb}GB memory, "
            f"but only {available_gb}GB available. "
            "Consider reducing data size or switching to CPU backend."
        )
        super().__init__(message)
