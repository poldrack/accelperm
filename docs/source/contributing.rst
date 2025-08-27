Contributing to AccelPerm
========================

We welcome contributions to AccelPerm! This document provides guidelines for contributing to the project.

Development Setup
-----------------

1. **Fork and Clone**::

    git clone https://github.com/your-username/accelperm.git
    cd accelperm

2. **Set up Development Environment**::

    # Using uv (recommended)
    uv sync --all-extras

    # Or using pip
    pip install -e ".[dev]"

3. **Install Pre-commit Hooks**::

    uv run pre-commit install

4. **Verify Installation**::

    uv run pytest
    uv run ruff check src/
    uv run mypy src/

Development Workflow
--------------------

1. **Create a Feature Branch**::

    git checkout -b feature/your-feature-name

2. **Follow TDD (Test-Driven Development)**:

   * Write failing tests first (RED)
   * Write minimal code to pass tests (GREEN)
   * Refactor and optimize (REFACTOR)

3. **Write Code**:

   * Follow PEP 8 style guidelines
   * Use comprehensive type hints
   * Write NumPy-style docstrings
   * Keep functions under 50 lines

4. **Run Tests**::

    uv run pytest
    uv run pytest --cov=accelperm --cov-report=html

5. **Check Code Quality**::

    uv run ruff check src/ tests/
    uv run ruff format src/ tests/
    uv run mypy src/

6. **Update Documentation** (if needed)::

    cd docs
    uv run sphinx-build -b html source _build/html

7. **Commit Changes**::

    git add .
    git commit -m "feat: add new feature description"

8. **Push and Create Pull Request**::

    git push origin feature/your-feature-name

Code Standards
--------------

**Python Style**

* Follow PEP 8
* Use ruff for linting and formatting
* Maximum line length: 88 characters
* Use meaningful variable names
* Prefer explicit over implicit

**Type Hints**

* Use comprehensive type hints for all public APIs
* Import from ``typing`` when needed
* Use ``from __future__ import annotations`` for forward references

Example::

    from __future__ import annotations

    import numpy as np
    from typing import Optional, Union, Tuple

    def compute_glm(
        y: np.ndarray,
        x: np.ndarray,
        contrasts: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute General Linear Model.

        Parameters
        ----------
        y : np.ndarray
            Response data of shape (n_observations, n_voxels)
        x : np.ndarray
            Design matrix of shape (n_observations, n_predictors)
        contrasts : np.ndarray, optional
            Contrast matrix of shape (n_contrasts, n_predictors)

        Returns
        -------
        beta : np.ndarray
            Parameter estimates
        residuals : np.ndarray
            Model residuals
        """

**Documentation**

* Use NumPy-style docstrings for all public functions
* Include parameter types and descriptions
* Include return value descriptions
* Add examples for complex functions
* Document any mathematical formulations

**Testing**

* Write unit tests for all new functions
* Achieve >90% test coverage
* Use descriptive test names
* Test edge cases and error conditions
* Use pytest fixtures for setup/teardown

Example test structure::

    def test_compute_glm_basic():
        """Test GLM computation with simple data."""
        # Arrange
        y = np.random.randn(100, 1000)
        x = np.random.randn(100, 3)

        # Act
        beta, residuals = compute_glm(y, x)

        # Assert
        assert beta.shape == (3, 1000)
        assert residuals.shape == (100, 1000)

Architecture Guidelines
-----------------------

**Backend Abstraction**

New backends must inherit from the ``Backend`` abstract base class::

    from abc import ABC, abstractmethod

    class NewBackend(Backend):
        @abstractmethod
        def compute_glm(self, y, x, contrasts):
            """Implement GLM computation for this backend."""
            pass

**Error Handling**

* Use specific exception classes
* Provide informative error messages
* Include suggestions for fixing errors

Example::

    class InsufficientGPUMemoryError(Exception):
        """Raised when GPU memory is insufficient for operation."""

        def __init__(self, required: float, available: float):
            super().__init__(
                f"Operation requires {required:.1f}GB GPU memory, "
                f"but only {available:.1f}GB available. "
                f"Consider reducing batch size or using CPU backend."
            )

**Performance Considerations**

* Profile before optimizing
* Use appropriate data types (float32 vs float64)
* Minimize memory allocations
* Cache expensive computations
* Implement chunking for large datasets

Pull Request Guidelines
-----------------------

**Before Submitting**

- [ ] Tests pass: ``uv run pytest``
- [ ] Linting passes: ``uv run ruff check src/ tests/``
- [ ] Type checking passes: ``uv run mypy src/``
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated for significant changes
- [ ] Performance tested if relevant

**Pull Request Template**

Use this template for your pull requests:

.. code-block:: markdown

    ## Description
    Brief description of the changes

    ## Type of Change
    - [ ] Bug fix (non-breaking change which fixes an issue)
    - [ ] New feature (non-breaking change which adds functionality)
    - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
    - [ ] Documentation update

    ## Testing
    - [ ] Unit tests added/updated
    - [ ] Integration tests added/updated
    - [ ] Manual testing performed

    ## Performance Impact
    - [ ] No performance impact
    - [ ] Performance improvement (include benchmarks)
    - [ ] Performance regression (justified and documented)

    ## Checklist
    - [ ] Code follows style guidelines
    - [ ] Self-review completed
    - [ ] Documentation updated
    - [ ] Tests added and passing

Statistical Validation
----------------------

For changes affecting statistical computations:

1. **Compare with FSL randomise** on test datasets
2. **Verify statistical properties** (null distributions, p-values)
3. **Test edge cases** (small samples, degenerate designs)
4. **Document mathematical changes** thoroughly

Example validation::

    def test_statistical_accuracy():
        """Validate against FSL randomise output."""
        # Load FSL reference results
        fsl_tstat = load_reference_data("fsl_tstat.nii.gz")

        # Compute with our implementation
        our_tstat = compute_tstat(data, design, contrasts)

        # Check accuracy
        relative_error = np.abs(our_tstat - fsl_tstat) / np.abs(fsl_tstat)
        assert np.max(relative_error) < 0.00001  # <0.001% error

Reporting Issues
----------------

When reporting bugs, please include:

1. **Environment Information**:

   * Python version
   * AccelPerm version
   * Operating system
   * GPU information (if relevant)

2. **Minimal Reproducible Example**:

   * Smallest possible dataset
   * Complete command or code
   * Expected vs actual behavior

3. **Error Messages**: Full traceback if available

4. **Context**: What you were trying to accomplish

Feature Requests
----------------

For feature requests:

1. **Describe the problem** your feature would solve
2. **Proposed solution** with examples
3. **Alternative solutions** you've considered
4. **Implementation considerations** (performance, complexity)

Community Guidelines
--------------------

* Be respectful and inclusive
* Help newcomers learn the codebase
* Focus on constructive feedback
* Follow the code of conduct
* Ask questions if anything is unclear

Recognition
-----------

Contributors will be acknowledged in:

* ``CONTRIBUTORS.md`` file
* Release notes for significant contributions
* Documentation for substantial improvements

Thank you for contributing to AccelPerm!
