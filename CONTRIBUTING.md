# Contributing to AccelPerm

We welcome contributions to AccelPerm! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/poldrack/accelperm.git
   cd accelperm
   ```

2. **Install uv** (recommended package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create virtual environment and install dependencies**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Unix
   # or
   .venv\Scripts\activate     # On Windows

   uv pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Before Making Changes

1. **Check TASKS.md** to see what needs to be done
2. **Read CLAUDE.md** for development guidelines
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Test-Driven Development

We follow strict TDD principles:

1. **Write failing tests first** (RED phase)
2. **Write minimal code to pass** (GREEN phase)
3. **Refactor and optimize** (REFACTOR phase)

### Code Quality Standards

- **Type hints**: All public APIs must have type hints
- **Docstrings**: NumPy-style docstrings for all public functions
- **Line length**: Maximum 88 characters
- **Test coverage**: >90% for unit tests, >80% for integration tests

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/unit/test_statistics.py

# Run benchmarks
make benchmark
```

### Code Formatting and Linting

```bash
# Format code
make format

# Run linting
make lint

# Run type checking
make type-check
```

### Documentation

- Update docstrings when changing function signatures
- Add examples to docstrings for complex functions
- Update user documentation for new features

## Statistical Validation

All statistical implementations must be validated against FSL randomise with <0.001% deviation. When implementing new statistical methods:

1. Create validation tests in `tests/validation/`
2. Compare results with FSL randomise on known datasets
3. Document any discrepancies or limitations

## Performance Requirements

GPU implementations must achieve minimum 10x speedup over CPU. When optimizing:

1. Profile before optimizing
2. Add performance benchmarks
3. Test on realistic dataset sizes
4. Document performance characteristics

## Pull Request Process

1. **Ensure tests pass**: All tests must pass before submitting PR
2. **Maintain coverage**: Test coverage should not decrease
3. **Update documentation**: Update relevant documentation
4. **Performance check**: No significant performance regressions
5. **Review checklist**:
   - [ ] Tests pass
   - [ ] Coverage maintained
   - [ ] Documentation updated
   - [ ] Performance benchmarked
   - [ ] Type hints complete

## GPU Development Guidelines

### Memory Management
- Always clean up GPU memory
- Use memory pooling for frequent allocations
- Handle out-of-memory errors gracefully
- Test with various GPU memory sizes

### Cross-Platform Compatibility
- Test on both CUDA and MPS backends
- Ensure CPU fallback works correctly
- Handle hardware detection properly

## Reporting Issues

When reporting bugs or requesting features:

1. **Search existing issues** first
2. **Use issue templates** when available
3. **Provide minimal reproduction** for bugs
4. **Include system information** (OS, Python version, GPU type)

## Communication

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Use GitHub Issues for bugs and feature requests
- **Email**: Contact maintainers for security issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- CHANGELOG.md
- AUTHORS file
- Documentation acknowledgments

Thank you for contributing to AccelPerm!
