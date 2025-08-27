"""Tests for the core orchestrator - TDD RED phase."""


from accelperm.core.orchestrator import Orchestrator


class TestOrchestrator:
    """Test the Orchestrator class - workflow coordinator."""

    def test_orchestrator_can_be_instantiated(self):
        """Test that Orchestrator can be instantiated - RED phase."""
        # This should fail because Orchestrator doesn't exist yet
        orchestrator = Orchestrator()
        assert orchestrator is not None

    def test_orchestrator_has_run_method(self):
        """Test that Orchestrator has a run method - RED phase."""
        # This should fail because the run method doesn't exist yet
        orchestrator = Orchestrator()
        assert hasattr(orchestrator, "run")

    def test_orchestrator_accepts_backend_parameter(self):
        """Test that Orchestrator accepts backend parameter - RED phase."""
        # This should fail because Orchestrator doesn't accept backend yet
        from accelperm.backends.base import Backend

        class MockBackend(Backend):
            def is_available(self):
                return True

            def compute_glm(self, Y, X, contrasts):
                del Y, X, contrasts  # Unused in mock
                return {}

        backend = MockBackend()
        orchestrator = Orchestrator(backend=backend)
        assert orchestrator.backend is backend
