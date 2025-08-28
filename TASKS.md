# TASKS.md - GPU-Accelerated Permutation Testing Task List

## Task Tracking Guidelines
- Mark completed tasks with [x]
- Add completion date in format: `[x] Task name (2024-01-15)`
- Add new discovered tasks under appropriate milestone
- If a task is blocked, add `[BLOCKED: reason]`
- If a task is in progress, add `[IN PROGRESS]`

---

## Phase 1: Foundation (Weeks 1-4)
**Goal**: Establish project infrastructure and core architecture

### Week 1: Project Setup and Infrastructure
- [x] Initialize Git repository with .gitignore for Python (completed)
- [x] Create project structure as defined in CLAUDE.md (completed)
  - [x] Create src/accelperm directory structure (completed)
  - [x] Create tests/ directory structure (completed)
  - [x] Create benchmarks/ directory (completed)
  - [x] Create docs/ directory (completed)
  - [x] Create examples/ directory (completed)
- [x] Set up pyproject.toml with uv (completed)
  - [x] Define project metadata (completed)
  - [x] Add core dependencies (completed)
  - [x] Add development dependencies (completed)
  - [x] Configure build system (completed)
- [x] Configure development tools (completed)
  - [x] Set up ruff configuration (completed)
  - [x] Set up mypy configuration (completed)
  - [x] Set up pytest configuration (completed)
  - [x] Create .pre-commit-config.yaml (completed)
- [x] Set up CI/CD pipeline (2025-08-27)
  - [x] Create GitHub Actions workflow for tests (2025-08-27)
  - [x] Add matrix testing for Python 3.12 and 3.13 (2025-08-27)
  - [x] Configure coverage reporting with Codecov (2025-08-27)
  - [x] Set up benchmark workflow (2025-08-27)
- [x] Create initial documentation files (completed)
  - [x] Write README.md with installation instructions (completed)
  - [x] Create CONTRIBUTING.md (completed)
  - [x] Create CHANGELOG.md (completed)
  - [x] Set up Sphinx documentation structure (completed)

### Week 2: Core Architecture Implementation
- [x] Implement backend abstraction layer (completed)
  - [x] Create src/accelperm/backends/base.py with ABC (completed)
  - [x] Define Backend abstract class interface (completed)
  - [x] Write unit tests for backend interface (completed)
- [x] Create core orchestrator module (completed)
  - [x] Implement src/accelperm/core/orchestrator.py skeleton (completed)
  - [x] Define workflow coordination interface (completed)
  - [x] Write unit tests for orchestrator (completed)
- [x] Set up logging framework (2025-08-27)
  - [x] Create consistent logging configuration (2025-08-27)
  - [x] Implement progress reporting interface (2025-08-27)
  - [x] Add debug logging capabilities (2025-08-27)
- [x] Implement configuration management (2025-08-27)
  - [x] Create settings module with Pydantic (2025-08-27)
  - [x] Define configuration schema (2025-08-27)
  - [x] Support config file loading (TOML) (2025-08-27)
- [x] Create error handling framework (2025-08-27)
  - [x] Define custom exception classes (2025-08-27)
  - [x] Implement error recovery strategies (2025-08-27)
  - [x] Add informative error messages (2025-08-27)

### Week 3: Data I/O Layer
- [x] Implement NIfTI file handling (2025-08-27)
  - [x] Create src/accelperm/io/nifti.py (2025-08-27)
  - [x] Implement NiftiLoader class with comprehensive functionality (2025-08-27)
  - [x] Implement load_nifti function with nibabel (2025-08-27)
  - [x] Implement save_nifti function (2025-08-27)
  - [x] Add support for compressed files (.nii.gz) (2025-08-27)
  - [x] Write comprehensive unit tests (2025-08-27)
  - [x] Add validation for file integrity (2025-08-27)
  - [x] Add memory optimization features (lazy loading, chunking) (2025-08-27)
  - [x] Add masking support for brain data (2025-08-27)
  - [x] Implement validate_nifti_compatibility function (2025-08-27)
- [x] Create design matrix loader (2025-08-27)
  - [x] Implement src/accelperm/io/design.py with DesignMatrixLoader class (2025-08-27)
  - [x] Support CSV format parsing with pandas backend (2025-08-27)
  - [x] Support TSV format parsing with pandas backend (2025-08-27)
  - [x] Add categorical variable encoding (one-hot encoding) (2025-08-27)
  - [x] Add intercept column support and standardization (2025-08-27)
  - [x] Validate design matrix structure (rank, missing values, constants) (2025-08-27)
  - [x] Write comprehensive unit tests with 17 tests, 100% pass rate (2025-08-27)
  - [x] Add utility functions: load_design_matrix, validate_design_matrix, create_contrast_matrix (2025-08-27)
  - [x] Add FSL and SPM format compatibility (2025-08-27)
  - [x] Implement Gram-Schmidt orthogonalization for correlated regressors (2025-08-27)
- [x] Implement contrast file loader (2025-08-27)
  - [x] Implement src/accelperm/io/contrast.py with ContrastLoader class (2025-08-27)
  - [x] Parse contrast matrices from text files (FSL .con, CSV, TXT formats) (2025-08-27)
  - [x] Validate contrast compatibility with design matrices (2025-08-27)
  - [x] Support multiple contrast files and batch loading (2025-08-27)
  - [x] Add contrast validation (rank deficiency, zero contrasts, data integrity) (2025-08-27)
  - [x] Add format compatibility (FSL, SPM styles) (2025-08-27)
  - [x] Add programmatic contrast creation (t-contrasts, F-contrasts, polynomial) (2025-08-27)
  - [x] Write comprehensive unit tests with 21 tests, 100% pass rate (2025-08-27)
  - [x] Add utility functions: load_contrast_matrix, validate_contrast_compatibility, create_t_contrast, create_f_contrast (2025-08-27)
- [x] Create output writer module (2025-08-27)
  - [x] Implement src/accelperm/io/output.py with OutputWriter class (2025-08-27)
  - [x] Support statistical map output to NIfTI format (2025-08-27)
  - [x] Support p-value map output to NIfTI format (2025-08-27)
  - [x] Support corrected p-value maps and TFCE maps (2025-08-27)
  - [x] Generate cluster tables in text format (2025-08-27)
  - [x] Add batch processing for multiple maps (2025-08-27)
  - [x] Add data validation (NaN/Inf checking, p-value ranges) (2025-08-27)
  - [x] Add format compatibility (FSL, SPM styles) (2025-08-27)
  - [x] Add analysis logging and metadata support (2025-08-27)
  - [x] Write comprehensive unit tests with 14 tests, 100% pass rate (2025-08-27)
  - [x] Add utility functions: save_statistical_map, save_p_value_map, generate_cluster_table, create_results_summary (2025-08-27)

### Week 4: CPU Backend and Basic GLM
- [x] Implement CPU backend (2025-08-27)
  - [x] Create src/accelperm/backends/cpu.py (2025-08-27)
  - [x] Implement CPUBackend class inheriting from Backend (2025-08-27)
  - [x] Use NumPy for array operations with neuroimaging-scale data support (2025-08-27)
  - [x] Write comprehensive unit tests for CPU backend - 11 tests, 90.48% coverage (2025-08-27)
- [x] Implement basic GLM in statistics module (2025-08-27)
  - [x] Create src/accelperm/core/statistics.py with GLMStatistics class (2025-08-27)
  - [x] Implement ordinary least squares (OLS) solver with pseudoinverse (2025-08-27)
  - [x] Calculate beta coefficients with numerical stability (2025-08-27)
  - [x] Compute residuals and variance with edge case handling (2025-08-27)
  - [x] Write unit tests against known results - 10 tests, 90.28% coverage (2025-08-27)
- [x] Implement t-statistic computation (2025-08-27)
  - [x] Calculate t-statistics from GLM with proper degrees of freedom (2025-08-27)
  - [x] Handle multiple contrasts with vectorized operations (2025-08-27)
  - [x] Validate against synthetic data with realistic neuroimaging dimensions (2025-08-27)
  - [x] Write comprehensive unit tests with numerical stability tests (2025-08-27)
- [x] Create initial CLI interface (2025-08-27)
  - [x] Implement src/accelperm/cli.py with Typer and Rich progress indicators (2025-08-27)
  - [x] Add comprehensive command-line arguments with validation (2025-08-27)
  - [x] Support input/output file specification with format validation (2025-08-27)
  - [x] Add extensive help documentation and error handling (2025-08-27)
  - [x] Write integration tests - 11 tests, 72.11% coverage (2025-08-27)

---

## Phase 2: GPU Acceleration (Weeks 5-8)
**Goal**: Implement GPU backends with MPS and CUDA support

### Week 5: MPS Backend Foundation
- [x] Set up PyTorch with MPS support (2025-08-27)
  - [x] Verify MPS availability detection (2025-08-27)
  - [x] Create device management utilities (2025-08-27)
  - [x] Handle MPS unavailability gracefully (2025-08-27)
- [x] Implement MPS backend structure (2025-08-27)
  - [x] Create src/accelperm/backends/mps.py (2025-08-27)
  - [x] Implement MPSBackend class (2025-08-27)
  - [x] Set up tensor operations on MPS device (2025-08-27)
  - [x] Write unit tests with comprehensive coverage (2025-08-27)
- [x] Port GLM to MPS (2025-08-27)
  - [x] Convert NumPy operations to PyTorch (2025-08-27)
  - [x] Implement tensor-based matrix operations (2025-08-27)
  - [x] Handle float32/float64 conversion for MPS (2025-08-27)
  - [x] Validate numerical accuracy vs CPU backend (2025-08-27)
- [x] Implement MPS memory management (2025-08-27)
  - [x] Create memory cleanup utilities (torch.mps.empty_cache) (2025-08-27)
  - [x] Implement CPU fallback for OOM errors (2025-08-27)
  - [x] Handle out-of-memory errors gracefully (2025-08-27)
  - [x] Write comprehensive test suite (17 tests, 100% pass rate) (2025-08-27)

**Week 5 Status**: COMPLETE ✅
**Key Achievements**:
- Full MPS backend implementation with PyTorch integration
- Comprehensive test coverage (17 tests, 100% pass rate, 82.14% coverage)
- Float32/float64 compatibility layer for MPS constraints
- CPU fallback mechanism for out-of-memory scenarios
- Device management with automatic availability detection
- CLI integration for MPS backend selection
- Numerical accuracy validation within acceptable tolerances

### Week 6: CUDA Backend Foundation
- [ ] Set up PyTorch with CUDA support
  - [ ] Implement CUDA availability detection
  - [ ] Query GPU capabilities
  - [ ] Handle multi-GPU systems
- [ ] Implement CUDA backend structure
  - [ ] Create src/gpu_randomise/backends/cuda.py
  - [ ] Implement CUDABackend class
  - [ ] Configure CUDA streams
  - [ ] Write unit tests with mocked CUDA
- [ ] Port GLM to CUDA
  - [ ] Implement CUDA-optimized operations
  - [ ] Use cuBLAS where applicable
  - [ ] Optimize grid and block dimensions
  - [ ] Validate against CPU results
- [ ] Implement CUDA-specific optimizations
  - [ ] Use pinned memory for transfers
  - [ ] Implement asynchronous operations
  - [ ] Overlap computation and transfer
  - [ ] Profile and optimize kernels

### Week 7: Backend Selection and Optimization
- [x] Implement intelligent backend selection (2025-08-28)
  - [x] Create backend factory with hardware detection (2025-08-28)
  - [x] Auto-detect available hardware (MPS, CPU) (2025-08-28)
  - [x] Estimate memory requirements for neuroimaging data (2025-08-28)
  - [x] Choose optimal backend based on data size (2025-08-28)
- [x] Create unified backend interface (2025-08-28)
  - [x] Ensure API consistency across backends (2025-08-28)
  - [x] Implement fallback mechanisms (2025-08-28)
  - [x] Add backend switching capability (2025-08-28)
  - [x] Write integration tests with 95% coverage (2025-08-28)
- [ ] Optimize data transfers [DEFERRED: To be addressed in performance optimization phase]
  - [ ] Minimize CPU-GPU transfers
  - [ ] Implement data prefetching
  - [ ] Use unified memory where available
  - [ ] Profile transfer overhead
- [x] Implement chunking for large datasets (2025-08-28)
  - [x] Calculate optimal chunk size based on memory (2025-08-28)
  - [x] Handle chunk boundary conditions correctly (2025-08-28)
  - [x] Maintain statistical validity across chunks (2025-08-28)
  - [x] Write comprehensive tests with various chunk sizes (2025-08-28)

**Week 7 Status**: 95% COMPLETE ✅ (Data transfer optimization deferred to Phase 4)
**Key Achievements**:
- Backend Factory: Intelligent hardware detection and selection
- ChunkedBackendWrapper: Memory-aware processing for large datasets
- Thread-safe backend caching with singleton pattern
- CLI integration with "auto" backend selection
- Comprehensive test coverage: 24 tests passing (85.71% factory, 95.35% chunking)

### Week 8: Performance Benchmarking
- [ ] Create benchmarking framework
  - [ ] Set up pytest-benchmark configuration
  - [ ] Define benchmark datasets
  - [ ] Create performance metrics
- [ ] Benchmark CPU backend
  - [ ] Measure GLM performance
  - [ ] Profile memory usage
  - [ ] Identify bottlenecks
- [ ] Benchmark MPS backend
  - [ ] Compare vs CPU performance
  - [ ] Measure GPU utilization
  - [ ] Profile memory patterns
- [ ] Benchmark CUDA backend
  - [ ] Compare vs CPU and MPS
  - [ ] Test multi-GPU scaling
  - [ ] Optimize based on results
- [ ] Create performance regression tests
  - [ ] Set baseline performance
  - [ ] Automate performance tracking
  - [ ] Add to CI pipeline

---

## Phase 3: Statistical Features (Weeks 9-12)
**Goal**: Implement permutation testing and multiple comparison corrections

### Week 9: Permutation Engine Core
- [x] Create permutation module structure (2025-08-28)
  - [x] Implement src/accelperm/core/permutation.py (2025-08-28)
  - [x] Define permutation strategy interface (ABC) (2025-08-28)
  - [x] Create permutation generator base class (2025-08-28)
- [x] Implement sign-flipping permutation (2025-08-28)
  - [x] Create SignFlippingStrategy for paired designs (2025-08-28)
  - [x] Generate permutation matrices with ±1 values (2025-08-28)
  - [x] Validate permutation properties (2025-08-28)
  - [x] Write comprehensive unit tests (31 tests passing) (2025-08-28)
- [x] Implement exchangeability blocks (2025-08-28)
  - [x] Parse exchangeability file (.grp) support (2025-08-28)
  - [x] Handle block-wise permutations (2025-08-28)
  - [x] Validate block constraints (2025-08-28)
  - [x] Write tests with various block structures (2025-08-28)
- [x] Create permutation scheduler (2025-08-28)
  - [x] Manage permutation batching (2025-08-28)
  - [x] Track permutation progress (2025-08-28)
  - [x] Implement checkpointing and restoration (2025-08-28)
  - [ ] Distribute across GPU cores (deferred to optimization phase)

**Week 9 Status**: 95% COMPLETE ✅
**Key Achievements**:
- Full permutation engine implementation with strategy pattern
- Sign-flipping and full permutation strategies
- Exchangeability blocks for restricted permutations
- Batch processing with progress tracking and checkpointing
- 31 comprehensive tests, 91.88% code coverage

### Week 10: Advanced Permutation Strategies ✅ COMPLETE
- [x] Implement adaptive Monte Carlo sampling
  - [x] Dynamic sample size adjustment based on test statistics variance
  - [x] Convergence detection using confidence intervals
  - [x] Batch processing with early stopping capabilities
  - [x] Write comprehensive tests (5 tests covering all scenarios)
- [x] Implement full enumeration detector
  - [x] Detect when full enumeration is feasible
  - [x] Memory requirement estimation with safety factors
  - [x] Automatic strategy selection (enumeration vs sampling)
  - [x] Write tests for feasibility detection (4 tests)
- [x] Implement variance smoothing
  - [x] Empirical Bayes shrinkage toward prior distributions
  - [x] Covariate-based smoothing within experimental groups
  - [x] Degrees of freedom adjustment for improved inference
  - [x] Write comprehensive unit tests (5 tests)
- [x] Implement two-stage permutation
  - [x] Between-subject and within-subject permutation handling
  - [x] Nested experimental structure support (subject/session/condition)
  - [x] Blocked randomization maintaining experimental balance
  - [x] Write tests for hierarchical designs (4 tests)
- [x] Create integration testing framework
  - [x] Cross-component validation and realistic workflows
  - [x] Test adaptive enumeration switching
  - [x] Test variance smoothing integration with permutation testing
  - [x] Write integration tests (3 comprehensive tests)

**Week 10 Summary:**
- Advanced permutation strategies module: `src/accelperm/core/advanced_permutation.py`
- Comprehensive test suite: `tests/unit/test_advanced_permutation.py`
- 21 tests total, 100% pass rate, 85.65% code coverage
- All methods designed for FSL randomise compatibility
- Statistical validation ensuring permutation properties are preserved

### Week 11: Multiple Comparison Corrections ✅ COMPLETE
- [x] Implement corrections module structure (2025-08-28)
  - [x] Create src/accelperm/core/corrections.py with comprehensive functionality (2025-08-28)
  - [x] Define CorrectionMethod ABC with validation and error handling (2025-08-28)
  - [x] Create CorrectionResult dataclass for standardized result handling (2025-08-28)
- [x] Implement Bonferroni correction (2025-08-28)
  - [x] Conservative FWER control with alpha/n adjustment (2025-08-28)
  - [x] Input validation for p-values and alpha parameters (2025-08-28)
  - [x] Write comprehensive unit tests (3 tests covering basic functionality and edge cases) (2025-08-28)
- [x] Implement FDR correction (2025-08-28)
  - [x] Benjamini-Hochberg procedure with proper step-up procedure (2025-08-28)
  - [x] Conservative mode option for dependent tests (2025-08-28)
  - [x] Adjusted p-values calculation with monotonicity preservation (2025-08-28)
  - [x] Write comprehensive unit tests (4 tests covering step-up procedure and conservative mode) (2025-08-28)
- [x] Implement FWER correction (2025-08-28)
  - [x] Max-statistic method using permutation null distributions (2025-08-28)
  - [x] Insufficient permutations warning system (2025-08-28)
  - [x] Write comprehensive unit tests (5 tests covering max-statistic method and permutation handling) (2025-08-28)
- [x] Implement cluster-based correction (2025-08-28)
  - [x] Spatial extent and mass correction with 3D connectivity options (6/18/26) (2025-08-28)
  - [x] Connected component detection using scipy.ndimage (2025-08-28)
  - [x] Cluster p-value calculation from null distributions (2025-08-28)
  - [x] Write comprehensive unit tests (5 tests covering 3D clustering and connectivity types) (2025-08-28)
- [x] Create integration and validation framework (2025-08-28)
  - [x] Cross-method comparison tests (3 integration tests) (2025-08-28)
  - [x] Realistic permutation test data validation (2025-08-28)
  - [x] Error handling and edge case validation (4 tests) (2025-08-28)
- [x] CLI integration for corrections (2025-08-28)
  - [x] Add correction parameters: --correction, --alpha, --n-perm (2025-08-28)
  - [x] Support for correction methods: none, bonferroni, fdr, fwer, cluster (2025-08-28)
  - [x] Enhanced output: corrected p-values, significance masks, detailed summaries (2025-08-28)

**Week 11 Summary:**
- Complete multiple comparison corrections module: `src/accelperm/core/corrections.py`
- Comprehensive test suite: `tests/unit/test_corrections.py`
- 27 tests total, 100% pass rate, 87.44% code coverage
- CLI integration with new correction parameters and enhanced output
- FSL randomise compatibility design patterns with proper statistical implementations

### Week 12: TFCE Implementation
- [ ] Research TFCE algorithm
  - [ ] Study Smith & Nichols (2009) paper
  - [ ] Examine FSL source code
  - [ ] Create implementation plan
- [ ] Implement basic TFCE
  - [ ] Create src/accelperm/core/tfce.py
  - [ ] Implement threshold stepping
  - [ ] Calculate cluster extent
  - [ ] Apply height and extent weighting
- [ ] Optimize TFCE for GPU
  - [ ] Parallelize threshold iterations
  - [ ] Optimize connected component labeling
  - [ ] Minimize memory usage
  - [ ] Profile performance
- [ ] Validate TFCE implementation
  - [ ] Compare with FSL randomise output
  - [ ] Test on known datasets
  - [ ] Verify statistical properties
  - [ ] Write comprehensive tests
- [ ] Integrate TFCE with corrections
  - [ ] Build null distributions for TFCE
  - [ ] Calculate empirical p-values
  - [ ] Apply TFCE-based correction method
  - [ ] Validate full pipeline

---

## Phase 4: Optimization & Polish (Weeks 13-16)
**Goal**: Optimize performance and improve user experience

### Week 13: Performance Optimization
- [ ] Profile entire pipeline
  - [ ] Identify performance bottlenecks
  - [ ] Measure GPU utilization
  - [ ] Analyze memory access patterns
  - [ ] Create optimization plan
- [ ] Optimize memory usage
  - [ ] Implement memory pooling
  - [ ] Reduce memory fragmentation
  - [ ] Optimize tensor allocation
  - [ ] Add memory profiling
- [ ] Optimize GPU kernels
  - [ ] Tune grid and block sizes
  - [ ] Optimize memory coalescing
  - [ ] Use shared memory effectively
  - [ ] Minimize divergent branches
- [ ] Implement mixed precision
  - [ ] Identify precision-sensitive operations
  - [ ] Use float32 where appropriate
  - [ ] Validate numerical stability
  - [ ] Measure performance impact

### Week 14: Statistical Validation
- [ ] Create validation test suite
  - [ ] Generate synthetic datasets with ground truth
  - [ ] Create known statistical scenarios
  - [ ] Define accuracy metrics
- [ ] Validate against FSL randomise
  - [ ] Run parallel tests with FSL
  - [ ] Compare statistical outputs
  - [ ] Verify <0.001% deviation
  - [ ] Document any differences
- [ ] Test edge cases
  - [ ] Single voxel datasets
  - [ ] Single subject analyses
  - [ ] Extreme statistical values
  - [ ] Degenerate design matrices
- [ ] Validate permutation distributions
  - [ ] Verify null distribution properties
  - [ ] Check p-value uniformity
  - [ ] Validate type I error rates
  - [ ] Test statistical power

### Week 15: CLI Enhancement and Compatibility
- [ ] Achieve FSL randomise CLI compatibility
  - [ ] Parse all randomise arguments
  - [ ] Map arguments to internal functions
  - [ ] Support legacy option formats
  - [ ] Write compatibility tests
- [ ] Add advanced CLI features
  - [ ] Progress bars with Rich
  - [ ] Verbose logging options
  - [ ] Dry-run capability
  - [ ] Configuration file support
- [ ] Implement batch processing
  - [ ] Support multiple input files
  - [ ] Parallel job execution
  - [ ] Job queuing system
  - [ ] Results aggregation
- [ ] Create GUI prototype (optional)
  - [ ] Design basic interface
  - [ ] Parameter configuration
  - [ ] Real-time progress monitoring
  - [ ] Results visualization

### Week 16: Documentation and Examples
- [ ] Write comprehensive user documentation
  - [ ] Installation guide for all platforms
  - [ ] Quick start tutorial
  - [ ] CLI reference manual
  - [ ] Migration guide from FSL
- [ ] Create API documentation
  - [ ] Document all public functions
  - [ ] Add code examples
  - [ ] Generate API reference with Sphinx
  - [ ] Add type hints documentation
- [ ] Develop example notebooks
  - [ ] Basic one-sample t-test example
  - [ ] Multiple regression example
  - [ ] TFCE demonstration
  - [ ] Performance comparison notebook
- [ ] Create test datasets
  - [ ] Generate small example datasets
  - [ ] Include real fMRI samples
  - [ ] Provide expected outputs
  - [ ] Document dataset properties

---

## Phase 5: Release Preparation (Weeks 17-18)
**Goal**: Prepare for public release

### Week 17: Testing and Bug Fixes
- [ ] Comprehensive integration testing
  - [ ] Test all permutation strategies
  - [ ] Test all correction methods
  - [ ] Test all backends
  - [ ] Test on various datasets
- [ ] Platform testing
  - [ ] Test on Linux (Ubuntu, CentOS)
  - [ ] Test on macOS (Intel and Apple Silicon)
  - [ ] Test on Windows
  - [ ] Document platform-specific issues
- [ ] Stress testing
  - [ ] Test with maximum data sizes
  - [ ] Test with minimal resources
  - [ ] Test error recovery
  - [ ] Test interruption handling
- [ ] Bug fixing sprint
  - [ ] Address all critical bugs
  - [ ] Fix documentation issues
  - [ ] Resolve compatibility problems
  - [ ] Update error messages

### Week 18: Release and Deployment
- [ ] Prepare release package
  - [ ] Update version numbers
  - [ ] Write release notes
  - [ ] Update CHANGELOG.md
  - [ ] Create release branch
- [ ] Set up package distribution
  - [ ] Configure PyPI packaging
  - [ ] Test package installation
  - [ ] Set up conda-forge recipe
  - [ ] Create Docker images
- [ ] Launch documentation
  - [ ] Deploy to ReadTheDocs
  - [ ] Set up versioned documentation
  - [ ] Create landing page
  - [ ] Add tutorials and guides
- [ ] Community preparation
  - [ ] Create GitHub issue templates
  - [ ] Set up discussion forum
  - [ ] Prepare announcement posts
  - [ ] Contact beta testers

---

## Continuous Tasks (Throughout Development)

### Code Quality
- [ ] Maintain >90% test coverage
- [ ] Keep all tests passing
- [ ] Address linting issues immediately
- [ ] Update type hints as needed
- [ ] Refactor when necessary

### Documentation
- [ ] Update docstrings with changes
- [ ] Keep README.md current
- [ ] Document design decisions
- [ ] Update user guides
- [ ] Maintain API changelog

### Performance
- [ ] Monitor performance regressions
- [ ] Track memory usage
- [ ] Optimize hot paths
- [ ] Profile regularly
- [ ] Benchmark against targets

### Communication
- [ ] Update SCRATCHPAD.md with notes
- [ ] Mark completed tasks promptly
- [ ] Add newly discovered tasks
- [ ] Document blockers
- [ ] Communicate progress

---

## Discovered Tasks (Add new tasks here as they arise)

### High Priority
- [ ] (Add discovered high-priority tasks here)

### Medium Priority
- [ ] (Add discovered medium-priority tasks here)

### Low Priority
- [ ] (Add discovered low-priority tasks here)

### Technical Debt
- [ ] (Add technical debt items here)

---

## Blocked Tasks

### Currently Blocked
- [ ] Task name [BLOCKED: reason]

---

## Completion Statistics

### Phase 1: Foundation
- Total tasks: 71 (Week 1: 15, Week 2: 14, Week 3: 28, Week 4: 14)
- Completed: 71 (100% - All weeks complete!)
- In Progress: 0
- Blocked: 0
- **Progress: 100%** ✅

### Phase 2: GPU Acceleration
- Total tasks: 36
- Completed: 30 (Week 5: 16/16, Week 7: 14/18)
- In Progress: 0
- Blocked: 0
- Deferred: 4 (data transfer optimizations)
- **Progress: 83.3%**

### Phase 3: Statistical Features
- Total tasks: 44  
- Completed: 44 (Week 9: 17/18, Week 10: 21/21, Week 11: 27/27)
- In Progress: 0 (Week 12 TFCE: 0/16)
- Blocked: 0
- **Progress: 75.0%**

### Phase 4: Optimization & Polish
- Total tasks: 52
- Completed: 0
- In Progress: 0
- Blocked: 0
- **Progress: 0%**

### Phase 5: Release Preparation
- Total tasks: 32
- Completed: 0
- In Progress: 0
- Blocked: 0
- **Progress: 0%**

### Week 3 Progress (Data I/O Layer) - **COMPLETE!** 🎉
- NIfTI handling: **COMPLETE** (10/10 subtasks)
- Design matrix loader: **COMPLETE** (10/10 subtasks)
- Contrast file loader: **COMPLETE** (10/10 subtasks)
- Output writer module: **COMPLETE** (12/12 subtasks)
- **Week 3 Progress: 100%** (42/42 subtasks complete)

### Overall Project
- **Total tasks: 262** (updated count)
- **Completed: 170 (64.9%)**
- **Phase 1: Foundation - COMPLETE** ✅
- **Phase 2: GPU Acceleration - 83% COMPLETE** (Week 5 MPS ✅, Week 7 Backend Selection ✅)
- **Phase 3: Statistical Features - 75% COMPLETE** (Week 9 Permutation Engine ✅, Week 10 Advanced Permutation ✅, Week 11 Multiple Comparison Corrections ✅)
- **Next: Week 12 - TFCE Implementation**

---

## Notes
- This task list is comprehensive but not exhaustive
- Add new tasks as they are discovered during development
- Some tasks may be split or combined based on actual complexity
- Priority may shift based on user feedback and requirements
- Update progress percentages after each work session

---

*Last Updated: 2025-08-28 (Week 11 Multiple Comparison Corrections Complete)*
*Next Review: 2025-09-04*
