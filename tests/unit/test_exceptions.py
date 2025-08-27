"""Tests for error handling framework - TDD RED phase."""


from accelperm.exceptions import (
    AccelPermError,
    BackendError,
    ConfigurationError,
    DataError,
    InsufficientMemoryError,
)


class TestAccelPermExceptions:
    """Test the error handling framework."""

    def test_base_exception_exists(self):
        """Test that AccelPermError base exception exists - RED phase."""
        # This should fail because AccelPermError doesn't exist yet
        error = AccelPermError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_backend_error_exists(self):
        """Test that BackendError exists - RED phase."""
        # This should fail because BackendError doesn't exist yet
        error = BackendError("Backend failed")
        assert str(error) == "Backend failed"
        assert isinstance(error, AccelPermError)

    def test_data_error_exists(self):
        """Test that DataError exists - RED phase."""
        # This should fail because DataError doesn't exist yet
        error = DataError("Invalid data format")
        assert str(error) == "Invalid data format"
        assert isinstance(error, AccelPermError)

    def test_configuration_error_exists(self):
        """Test that ConfigurationError exists - RED phase."""
        # This should fail because ConfigurationError doesn't exist yet
        error = ConfigurationError("Invalid configuration")
        assert str(error) == "Invalid configuration"
        assert isinstance(error, AccelPermError)

    def test_insufficient_memory_error_with_details(self):
        """Test InsufficientMemoryError with memory details - RED phase."""
        # This should fail because InsufficientMemoryError doesn't exist yet
        error = InsufficientMemoryError(required_gb=8.0, available_gb=4.0)
        expected_msg = (
            "Operation requires 8.0GB memory, but only 4.0GB available. "
            "Consider reducing data size or switching to CPU backend."
        )
        assert str(error) == expected_msg
        assert error.required_gb == 8.0
        assert error.available_gb == 4.0

    def test_backend_error_with_suggestion(self):
        """Test BackendError includes recovery suggestions - RED phase."""
        # This should fail because enhanced BackendError doesn't exist yet
        error = BackendError(
            "CUDA backend failed",
            suggestion="Try switching to CPU backend with: backend='cpu'",
        )
        expected_msg = (
            "CUDA backend failed\n"
            "Suggestion: Try switching to CPU backend with: backend='cpu'"
        )
        assert str(error) == expected_msg
        assert error.suggestion == "Try switching to CPU backend with: backend='cpu'"
