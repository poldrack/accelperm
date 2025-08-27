"""Tests for the backend abstraction layer - TDD approach."""

import numpy as np
import pytest

from accelperm.backends.base import Backend


class MockBackend(Backend):
    """Minimal mock backend for testing."""

    def is_available(self) -> bool:
        return True

    def compute_glm(self, Y, X, contrasts):
        del contrasts  # Unused in mock
        return {"beta": np.zeros((X.shape[1], Y.shape[0]))}


class TestBackend:
    """Test the Backend abstract base class."""

    def test_backend_cannot_be_instantiated_directly(self):
        """Test that Backend is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Backend()

    def test_concrete_backend_can_be_instantiated(self):
        """Test that a concrete backend implementation can be instantiated."""
        backend = MockBackend()
        assert backend.is_available() is True

    def test_backend_requires_compute_glm_method(self):
        """Test that Backend requires compute_glm method - RED phase."""
        # This should fail because MockBackend doesn't implement compute_glm
        with pytest.raises(TypeError):

            class IncompleteBackend(Backend):
                def is_available(self):
                    return True

                # Missing compute_glm method

            IncompleteBackend()

    def test_backend_requires_apply_permutation_method(self):
        """Test that Backend requires apply_permutation method."""
        # This will help us add apply_permutation as abstract method
        pass  # Will implement after adding the method
