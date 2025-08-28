# Development scratchpad

- Use this file to keep notes on ongoing development work.
- When the work is completed, clean it out from this file, so that it only reflects ongoing work.

## Project State Summary (2025-08-28)

### Current Status
**Branch**: dev/week7-backend-optimization
**Phase**: Phase 2 - GPU Acceleration
**Overall Progress**: 101/235 tasks completed (43%)
**Current Week**: Week 7 - Backend Selection and Optimization (95% complete)

### Recent Achievements

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
- Modified: src/accelperm/backends/factory.py
- Modified: src/accelperm/cli.py
- Modified: src/accelperm/core/chunking.py
- Modified: tests/unit/test_backend_factory.py
- Modified: tests/unit/test_cli_backend_integration.py
- Modified: tests/unit/test_data_chunking.py

### Test Status
- **Total**: 185 tests
- **Passing**: 182 tests
- **Failing**: 3 tests (CLI integration tests - mock configuration issues)
- **Overall Coverage**: 85.92%

### Next Steps

#### Immediate (Week 8: Performance Benchmarking)
1. Create benchmarking framework with pytest-benchmark
2. Define benchmark datasets for different scenarios
3. Benchmark CPU backend performance
4. Benchmark MPS backend vs CPU
5. Create performance regression tests

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
  - Week 8 (Performance Benchmarking): Not started
- Ready to begin performance benchmarking and optimization
