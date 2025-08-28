# Development scratchpad

- Use this file to keep notes on ongoing development work.
- When the work is completed, clean it out from this file, so that it only reflects ongoing work.

## Project State Summary (2025-08-28)

### Current Status
**Branch**: dev/week7-backend-optimization
**Phase**: Phase 3 - Statistical Features
**Overall Progress**: 143/235 tasks completed (60.9%)
**Current Week**: Week 10 - Advanced Permutation Strategies (100% complete)

### Recent Achievements

#### Week 10: Advanced Permutation Strategies (100% Complete)
- ✅ **AdaptiveMonteCarlo class** - Intelligent Monte Carlo sampling with convergence detection
  - Dynamic sample size adjustment based on test statistics variance
  - P-value convergence detection using confidence intervals
  - Batch processing with early stopping capabilities
  - 5 comprehensive tests with edge case coverage

- ✅ **FullEnumerationDetector class** - Automatic strategy selection for permutation testing
  - Feasibility detection based on memory and computational constraints
  - Memory requirement estimation with safety factors
  - Automatic switching between full enumeration and Monte Carlo sampling
  - 4 tests covering small/large sample scenarios

- ✅ **VarianceSmoothing class** - Advanced variance estimation for improved statistical power
  - Empirical Bayes shrinkage toward prior distributions
  - Covariate-based smoothing within experimental groups
  - Degrees of freedom adjustment for improved inference
  - 5 tests covering multiple smoothing methods

- ✅ **TwoStagePermutation class** - Complex hierarchical design support
  - Between-subject and within-subject permutation handling
  - Nested experimental structure support (subject/session/condition)
  - Blocked randomization maintaining experimental balance
  - 4 tests covering realistic neuroimaging experimental designs

- ✅ **Integration Testing** - Cross-component validation and realistic workflows
  - Adaptive enumeration switching based on problem characteristics
  - Variance smoothing integration with permutation testing
  - Two-stage permutation with Monte Carlo sampling
  - 3 comprehensive integration tests

#### Week 7: Backend Selection and Optimization (95% Complete)
- ✅ **BackendFactory class** - Intelligent hardware detection and backend selection
  - Auto-detection of MPS, CPU capabilities
  - Memory requirement estimation for neuroimaging data dimensions
  - Optimal backend selection based on data size and available hardware
  - Thread-safe backend caching with singleton pattern
  - 13 tests, 85.71% coverage

- ✅ **ChunkedBackendWrapper** - Memory-aware processing for large datasets
  - Automatic chunk size calculation based on memory constraints
  - Transparent chunking of any backend implementation
  - Statistical validity maintained across chunk boundaries
  - Result reconstruction from chunked processing
  - 11 tests, 95.35% coverage

- ✅ **CLI Integration** - Backend selection integrated into CLI
  - Support for "auto" backend selection
  - Manual backend override options (--backend cpu/mps/auto)
  - Progress reporting for chunked operations

#### Week 5: MPS Backend Foundation (Complete)
- ✅ Full MPS backend implementation with PyTorch
- ✅ Device management and automatic MPS detection
- ✅ Memory management with torch.mps.empty_cache()
- ✅ Float32/float64 compatibility layer
- ✅ CPU fallback mechanism for OOM scenarios
- ✅ 17 tests, 82.14% coverage

#### Week 4: CPU Backend and CLI (Complete)
- ✅ CPU backend with NumPy/SciPy GLM computation
- ✅ Statistics module with OLS, t-statistics, F-statistics
- ✅ CLI interface using Typer with Rich progress indicators
- ✅ 141 tests passing with 86.28% overall coverage

### Current Git Status
- Clean working directory after Week 10 commit
- New files committed:
  - `src/accelperm/core/advanced_permutation.py` (advanced permutation strategies)
  - `tests/unit/test_advanced_permutation.py` (comprehensive test suite)
- Previous modifications still present:
  - Backend factory, CLI, and data chunking improvements from Week 7

### Test Status
- **Total**: 237 tests
- **Passing**: 234 tests (98.7% pass rate)
- **Failing**: 3 tests (CLI integration tests - mock configuration issues, pre-existing)
- **Overall Coverage**: 86.80%
- **Advanced Permutation Module**: 85.65% coverage (21 tests, 100% pass rate)

### Next Steps

#### Immediate (Week 11: Multiple Comparison Corrections)
1. Implement Family-Wise Error Rate (FWER) correction
2. Implement False Discovery Rate (FDR) correction
3. Implement cluster-based correction methods
4. Implement voxel-wise correction methods
5. Create comprehensive test suite for correction methods

**Note**: `src/accelperm/core/corrections.py` exists but is empty (placeholder)

#### Short-term
1. Fix remaining CLI integration test failures (mocking issues)
2. Implement CUDA backend (Week 6 - currently skipped)
3. Complete data transfer optimizations (deferred from Week 7)
4. Begin Phase 3: Statistical Features (Week 9+)

### Technical Notes

#### Backend Factory Design
- Uses singleton pattern for thread-safe backend caching
- Estimates memory requirements: matrices + intermediate results
- Prefers GPU when available and data fits in memory
- Falls back to CPU for small datasets or memory constraints

#### Chunking Implementation
- Calculates chunk size based on available memory (80% threshold)
- Ensures minimum chunk size of 100 voxels
- Preserves statistical accuracy across chunk boundaries
- Transparent wrapper works with any backend

#### Performance Considerations
- Current MPS implementation prioritizes correctness over speed
- CPU-GPU transfers and scipy.stats calls limit performance
- Performance optimizations planned for Phase 4
- Chunking prevents OOM errors on large datasets

### Known Issues
1. **CLI Integration Tests**: 3 tests failing due to mock configuration
   - Need to refine mocking strategy for BackendFactory in CLI tests
   - Core functionality works, issue is with test setup

2. **Performance**: MPS backend not yet optimized
   - Currently uses scipy.stats for distributions (CPU fallback)
   - Data transfers between CPU/GPU not optimized
   - Will address in performance optimization phase

### Architecture Decisions
1. **Backend Selection**: Automatic by default, manual override available
2. **Memory Management**: Conservative 80% memory usage threshold
3. **Chunking**: Transparent to user, maintains statistical validity
4. **Error Handling**: Graceful degradation from GPU to CPU on failures

### Summary
- Phase 1 (Foundation): 100% complete ✅
- Phase 2 (GPU Acceleration): 83% complete
  - Week 5 (MPS Backend): Complete ✅
  - Week 6 (CUDA Backend): Not started (skipped)
  - Week 7 (Backend Selection): 95% complete ✅
  - Week 8 (Performance Benchmarking): Not started (deferred)
- Phase 3 (Statistical Features): 50% complete
  - Week 9 (Permutation Engine Core): Complete ✅
  - Week 10 (Advanced Permutation Strategies): Complete ✅
  - Week 11 (Multiple Comparison Corrections): Ready to start
  - Week 12 (TFCE Implementation): Not started (placeholder file exists)
- Ready to begin multiple comparison corrections implementation
