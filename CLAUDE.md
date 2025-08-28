# CLAUDE.md - Development Guide for GPU-Accelerated Permutation Testing

## Project Overview

You are helping develop a high-performance Python application for GPU-accelerated permutation testing on fMRI neuroimaging data. This is inspired by FSL's randomise program but leverages modern GPU architectures (CUDA and Apple Metal Performance Shaders) for 10-100x performance improvements.

**Key Goal**: Build a statistically accurate, GPU-accelerated alternative to FSL randomise that maintains <0.001% deviation from reference implementation while achieving >10x speedup.

## Workflow guidelines

- Always read PLANNING.md at the start of every new conversation
- Check TASKS.md and SCRATCHPAD.md before starting your work
- Mark completed tasks immediately within TASKS.md
- Add newly discovered tasks to TASKS.md
- use SCRATCHPAD.md as a scratchpad to outline plans

## FSL Implementation Reference

**IMPORTANT**: Use the detailed pseudocode in `PSEUDOCODE.md` as the authoritative reference when implementing FSL-compatible components. This file contains comprehensive pseudocode extracted from the FSL randomise source code, including:

- Complete permutation engine algorithms (sign-flipping vs. full permutation)
- Exact GLM T-statistic and F-statistic calculations
- TFCE implementation with optimized connected components
- Multiple comparison corrections (cluster, TFCE, FDR)
- Confound handling methods (Kennedy, Freedman-Lane, ter Braak)
- All utility functions for statistical compatibility

When implementing any statistical function, algorithm, or workflow component, **always reference the corresponding pseudocode section first** to ensure exact compatibility with FSL randomise. The pseudocode provides the exact mathematical formulations, parameter handling, and edge cases that must be preserved for <0.001% deviation from reference implementation.


## Current Development Phase

Track progress in this section. Update after completing each task.

**Current Phase**: Statistical Features (Weeks 9-12) - Phase 3
- [x] Week 9: Permutation Engine Core - COMPLETE ✅
- [x] Week 10: Advanced Permutation Strategies - COMPLETE ✅
- [ ] Week 11: Multiple Comparison Corrections - IN PROGRESS
- [ ] Week 12: TFCE Implementation - PENDING

**Previous Phases COMPLETE:**
- [x] Phase 1: Foundation (Weeks 1-4) - 100% ✅
- [x] Phase 2: GPU Acceleration (Weeks 5-8) - 83% ✅ (Week 6 & 8 deferred)

## Critical Development Principles

### 0. Package Management
**IMPORTANT**: This project uses `uv` for all Python operations. ALWAYS use `uv run` to execute commands:
```bash
# Run tests
uv run pytest

# Run linting
uv run ruff check

# Run type checking
uv run mypy

# Install dependencies
uv sync

# Any Python script execution
uv run python script.py
```

## Critical Development Principles

### 1. Test-Driven Development (TDD)
**CRITICAL**: This project STRICTLY follows TDD. NO exceptions.

**ALWAYS follow this cycle:**
1. **RED**: Write failing tests FIRST
   - Write tests before ANY implementation code
   - Tests MUST fail initially (red state)
   - If tests pass immediately, you violated TDD
2. **GREEN**: Write minimal code to pass tests
   - Implement ONLY enough to make tests pass
   - Resist the urge to over-engineer
   - Focus on making tests green, not perfect code
3. **REFACTOR**: Optimize and clean up
   - Improve code quality while keeping tests green
   - Apply design patterns and optimizations
   - Maintain test coverage throughout

**TDD Violations Are Not Acceptable:**
- Writing implementation before tests
- Writing tests that pass immediately
- Skipping the red phase
- Over-implementing in the green phase

**Why TDD is Critical for This Project:**
- Ensures statistical accuracy through comprehensive testing
- Provides confidence when optimizing performance-critical code
- Catches regressions early in GPU acceleration work
- Documents expected behavior for complex algorithms
- Enables safe refactoring of mathematical computations

**TDD Enforcement:**
- Every new feature starts with a failing test
- Code reviews must verify TDD was followed
- No merges without proper test-first development

### 2. Code Quality Standards
- Python 3.12+ with comprehensive type hints
- PEP 8 compliance (enforce with ruff)
- NumPy-style docstrings for all public APIs
- Maximum function length: 50 lines (refactor if longer)
- Test coverage: >90% unit tests, >80% integration tests

### 3. Architecture Patterns
- **Strategy pattern** for backend selection (CPU/CUDA/MPS)
- **Factory pattern** for data loader creation
- **Observer pattern** for progress reporting
- **Dependency injection** for testability

## System Architecture

```
┌─────────────────────────────────────────────────┐
│                  CLI/GUI Interface              │
├─────────────────────────────────────────────────┤
│                   Core Engine                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │Statistics│ │Permutation│ │Correction│      │
│  │  Module  │ │  Engine   │ │  Module  │      │
│  └──────────┘ └──────────┘ └──────────┘      │
├─────────────────────────────────────────────────┤
│              Backend Abstraction Layer          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │   MPS    │ │    CUDA  │ │    CPU   │      │
│  │  Backend │ │  Backend  │ │  Backend │      │
│  └──────────┘ └──────────┘ └──────────┘      │
├─────────────────────────────────────────────────┤
│                 Data I/O Layer                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │  NIfTI   │ │   Design  │ │  Output   │      │
│  │  Loader  │ │   Loader  │ │  Writer   │      │
│  └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────┘
```

## Technology Stack

### Core Dependencies
```toml
[dependencies]
python = ">=3.12"
numpy = ">=1.24"
scipy = ">=1.11"
nibabel = ">=5.0"  # Neuroimaging I/O
torch = ">=2.0"    # GPU acceleration (CUDA + MPS)
```

### Development Dependencies
```toml
[dev-dependencies]
pytest = ">=7.0"
pytest-cov = ">=4.0"
pytest-benchmark = ">=4.0"
ruff = ">=0.1"
mypy = ">=1.0"
sphinx = ">=7.0"
```

## Project Structure

```
gpu-randomise/
├── src/
│   └── gpu_randomise/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── orchestrator.py    # Main workflow coordinator
│       │   ├── statistics.py      # GLM and test statistics
│       │   ├── permutation.py     # Permutation strategies
│       │   └── corrections.py     # Multiple comparison corrections
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py           # Abstract backend interface
│       │   ├── cpu.py            # NumPy/Numba implementation
│       │   ├── mps.py            # Apple Metal implementation
│       │   └── cuda.py           # NVIDIA CUDA implementation
│       ├── io/
│       │   ├── __init__.py
│       │   ├── nifti.py          # NIfTI file handling
│       │   ├── design.py         # Design matrix loading
│       │   └── output.py         # Results writing
│       └── cli.py                # Command-line interface
├── tests/
│   ├── unit/                     # Unit tests for each module
│   ├── integration/              # Integration tests
│   ├── validation/               # Statistical validation tests
│   └── fixtures/                 # Test data and fixtures
├── benchmarks/                    # Performance benchmarks
├── docs/                          # Documentation
├── examples/                      # Example notebooks
├── pyproject.toml
├── README.md
├── CLAUDE.md                      # This file
└── PRD.md                         # Project requirements
```

## Key Implementation Tasks

### Phase 1: Foundation (Current)

1. **Project Setup**
   ```python
   # Create pyproject.toml with uv
   # Setup pre-commit hooks for ruff, mypy
   # Configure pytest with coverage
   # Setup GitHub Actions CI
   ```

2. **Core Architecture**
   ```python
   # src/gpu_randomise/backends/base.py
   from abc import ABC, abstractmethod

   class Backend(ABC):
       @abstractmethod
       def compute_glm(self, Y, X, contrasts):
           """Compute GLM statistics"""
           pass

       @abstractmethod
       def permute_data(self, data, strategy):
           """Apply permutation strategy"""
           pass
   ```

3. **Basic NIfTI I/O**
   ```python
   # src/gpu_randomise/io/nifti.py
   import nibabel as nib

   def load_nifti(filepath):
       """Load NIfTI file and return data + header"""
       # Implementation with proper error handling
   ```

4. **CPU Backend with GLM**
   ```python
   # src/gpu_randomise/backends/cpu.py
   import numpy as np
   from scipy import stats

   class CPUBackend(Backend):
       def compute_glm(self, Y, X, contrasts):
           # Implement GLM using NumPy/SciPy
   ```

### Phase 2: GPU Acceleration (Weeks 5-8)

1. **MPS Backend** (Priority for Apple Silicon)
   ```python
   # src/gpu_randomise/backends/mps.py
   import torch

   class MPSBackend(Backend):
       def __init__(self):
           self.device = torch.device("mps")
   ```

2. **CUDA Backend** (After MPS)
   ```python
   # src/gpu_randomise/backends/cuda.py
   class CUDABackend(Backend):
       def __init__(self):
           self.device = torch.device("cuda")
   ```

### Phase 3: Statistical Features (Weeks 9-12)

1. **Permutation Strategies**
   - Sign-flipping for paired designs
   - Full enumeration for small samples
   - Monte Carlo sampling for large samples
   - Block permutation

2. **TFCE Implementation**
   - Threshold-free cluster enhancement
   - Critical for neuroimaging applications

3. **Multiple Comparison Corrections**
   - Family-wise error rate (FWER)
   - Voxel-wise correction
   - Cluster-based correction

## Statistical Validation Requirements

### Critical Accuracy Targets
- **Deviation from FSL randomise**: <0.001%
- **Test with known datasets**: Human Connectome Project samples
- **Synthetic data validation**: Generate data with known ground truth

### Validation Tests to Implement
```python
# tests/validation/test_statistical_accuracy.py
def test_glm_against_fsl():
    """Compare GLM results with FSL randomise output"""

def test_permutation_distribution():
    """Verify permutation null distribution"""

def test_tfce_implementation():
    """Validate TFCE against reference"""
```

## Performance Benchmarks

### Target Performance
- **Small dataset**: 1000 voxels, 20 subjects, 1000 permutations → <10 seconds
- **Large dataset**: 250,000 voxels, 100 subjects, 10000 permutations → <60 seconds
- **Speedup vs CPU**: Minimum 10x, target 50x+

### Benchmark Implementation
```python
# benchmarks/benchmark_backends.py
import pytest
from gpu_randomise.backends import CPUBackend, MPSBackend

@pytest.mark.benchmark
def test_backend_performance(benchmark):
    # Benchmark each backend
```

## FSL Randomise Compatibility

### Command-line Compatibility Examples
```bash
# One-sample t-test
randomise -i input.nii -o output -1 -T -c 3.0 -v 10

# With exchangeability blocks
randomise -i input.nii -o output -1 -T -c 3.0 -v 10 -e design.grp --permuteBlocks

# Multiple regression
randomise -i input.nii -o output -d design.mat -t design.con -T -c 3.0 -v 10
```

Our implementation should support equivalent functionality.

## Common Development Workflows

### Adding a New Statistical Test
1. Write comprehensive unit tests in `tests/unit/test_statistics.py`
2. Implement in `src/gpu_randomise/core/statistics.py`
3. Add backend-specific optimizations
4. Validate against FSL output
5. Benchmark performance

### Implementing a GPU Kernel
1. Start with CPU reference implementation
2. Write tests comparing CPU vs GPU outputs
3. Implement PyTorch version for MPS/CUDA
4. Profile and optimize memory access patterns
5. Add fallback for unsupported operations

### Debugging GPU Memory Issues
1. Use `torch.cuda.memory_summary()` for CUDA
2. Implement chunking for large datasets
3. Clear cache between permutations if needed
4. Monitor with `nvidia-smi` or Activity Monitor

## Error Handling Guidelines

```python
# Always provide informative error messages
class InsufficientGPUMemoryError(Exception):
    """Raised when GPU memory is insufficient for operation"""
    def __init__(self, required, available):
        super().__init__(
            f"Operation requires {required}GB GPU memory, "
            f"but only {available}GB available. "
            f"Consider reducing batch size or using CPU backend."
        )
```

## Testing Strategy

### Test Categories
1. **Unit Tests** (`tests/unit/`)
   - Test individual functions in isolation
   - Mock external dependencies
   - Fast execution (<1s per test)

2. **Integration Tests** (`tests/integration/`)
   - Test module interactions
   - Use real data files
   - Verify end-to-end workflows

3. **Statistical Tests** (`tests/validation/`)
   - Compare with FSL randomise
   - Verify statistical properties
   - Use synthetic and real datasets

4. **Performance Tests** (`benchmarks/`)
   - Track performance regressions
   - Compare backend speeds
   - Memory usage profiling

### Running Tests
```bash
# IMPORTANT: Always use uv to execute commands in this project
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=accelperm --cov-report=html

# Run only unit tests
uv run pytest tests/unit/

# Run benchmarks
uv run pytest benchmarks/ --benchmark-only
```

## Documentation Standards

### Docstring Example
```python
def compute_tfce(stat_map: np.ndarray,
                 dh: float = 0.1,
                 E: float = 0.5,
                 H: float = 2.0) -> np.ndarray:
    """
    Compute Threshold-Free Cluster Enhancement.

    Parameters
    ----------
    stat_map : np.ndarray
        Statistical map to enhance (voxels x 1)
    dh : float, default=0.1
        Step size for threshold increments
    E : float, default=0.5
        Extent exponent
    H : float, default=2.0
        Height exponent

    Returns
    -------
    np.ndarray
        TFCE-enhanced statistical map

    References
    ----------
    Smith & Nichols (2009) - Threshold-free cluster enhancement
    """
```

## Debugging FSL Randomise Behavior

When implementing features, examine FSL source code:
1. Check FSL's C++ implementation for algorithms
2. Generate test cases using FSL randomise
3. Log intermediate values for comparison
4. Use small datasets for easier debugging

## Performance Optimization Checklist

- [ ] Profile code to identify bottlenecks
- [ ] Minimize CPU-GPU data transfers
- [ ] Use appropriate data types (float32 vs float64)
- [ ] Implement batch processing for large datasets
- [ ] Cache repeated computations
- [ ] Use GPU-optimized libraries (cuBLAS, etc.)
- [ ] Implement memory pooling for allocations
- [ ] Consider mixed precision where appropriate

## Common Pitfalls to Avoid

1. **Don't assume GPU is always faster** - Small datasets may be faster on CPU
2. **Watch for numerical precision issues** - GPUs use different floating-point implementations
3. **Avoid excessive memory allocation** - Reuse buffers where possible
4. **Don't forget edge cases** - Test with 1 voxel, 1 subject, 1 permutation
5. **Remember platform differences** - Test on Linux, macOS, Windows

## Questions to Ask Before Implementation

1. What is the exact mathematical formula being implemented?
2. How does FSL randomise handle this case?
3. What are the memory requirements?
4. Can this be parallelized effectively?
5. What are the numerical stability concerns?
6. How will we validate correctness?
7. What's the expected performance improvement?

## Progress Tracking

Update this section after each work session:

### Completed
- [Date] Initial project structure created
- [Date] ...

### In Progress
- Task: ...
- Blockers: ...

### Next Steps
- Priority 1: ...
- Priority 2: ...

## References and Resources

- [FSL Randomise Documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/statistics/randomise)
- [Winkler et al. (2014)](https://doi.org/10.1016/j.neuroimage.2014.01.060) - Permutation inference for the GLM
- [Smith & Nichols (2009)](https://doi.org/10.1016/j.neuroimage.2008.03.061) - TFCE
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## Session Guidelines for Claude Code

When starting a new session:
1. Check the Current Development Phase section
2. Review Progress Tracking for context
3. Focus on the next uncompleted task
4. Follow TDD methodology strictly
5. Update this file with progress before ending session

Remember: Write tests first, implement minimal solution, then optimize!
