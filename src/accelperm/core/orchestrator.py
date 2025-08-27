"""Core orchestrator for coordinating permutation testing workflows."""

from accelperm.backends.base import Backend


class Orchestrator:
    """Orchestrator for managing the entire permutation testing pipeline."""

    def __init__(self, backend: Backend | None = None) -> None:
        """Initialize the orchestrator.

        Args:
            backend: The computational backend to use for operations.
        """
        self.backend = backend

    def run(self) -> None:
        """Run the permutation testing pipeline."""
        pass
