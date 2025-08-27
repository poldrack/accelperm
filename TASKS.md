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
- [ ] Implement contrast file loader
  - [ ] Parse contrast matrices from text files
  - [ ] Validate contrast compatibility with design
  - [ ] Support multiple contrasts
  - [ ] Write unit tests
- [ ] Create output writer module
  - [x] Implement src/accelperm/io/output.py (file created, needs implementation)
  - [ ] Support statistical map output
  - [ ] Support p-value map output
  - [ ] Generate cluster tables
  - [ ] Write unit tests

### Week 4: CPU Backend and Basic GLM
- [ ] Implement CPU backend
  - [x] Create src/accelperm/backends/cpu.py (file created, needs implementation)
  - [ ] Implement CPUBackend class inheriting from Backend
  - [ ] Use NumPy for array operations
  - [ ] Write unit tests for CPU backend
- [ ] Implement basic GLM in statistics module
  - [x] Create src/accelperm/core/statistics.py (file created, needs implementation)
  - [ ] Implement ordinary least squares (OLS) solver
  - [ ] Calculate beta coefficients
  - [ ] Compute residuals and variance
  - [ ] Write unit tests against known results
- [ ] Implement t-statistic computation
  - [ ] Calculate t-statistics from GLM
  - [ ] Handle multiple contrasts
  - [ ] Validate against synthetic data
  - [ ] Write unit tests
- [ ] Create initial CLI interface
  - [x] Implement src/accelperm/cli.py with Typer (file created, needs implementation)
  - [ ] Add basic command-line arguments
  - [ ] Support input/output file specification
  - [ ] Add help documentation
  - [ ] Write integration tests

---

## Phase 2: GPU Acceleration (Weeks 5-8)
**Goal**: Implement GPU backends with MPS and CUDA support

### Week 5: MPS Backend Foundation
- [ ] Set up PyTorch with MPS support
  - [ ] Verify MPS availability detection
  - [ ] Create device management utilities
  - [ ] Handle MPS unavailability gracefully
- [ ] Implement MPS backend structure
  - [ ] Create src/gpu_randomise/backends/mps.py
  - [ ] Implement MPSBackend class
  - [ ] Set up tensor operations on MPS device
  - [ ] Write unit tests with mocked MPS
- [ ] Port GLM to MPS
  - [ ] Convert NumPy operations to PyTorch
  - [ ] Implement batched matrix operations
  - [ ] Optimize memory transfers
  - [ ] Validate numerical accuracy
- [ ] Implement MPS memory management
  - [ ] Create memory pool allocator
  - [ ] Implement tensor caching
  - [ ] Handle out-of-memory errors
  - [ ] Write stress tests

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
- [ ] Implement intelligent backend selection
  - [ ] Create backend factory
  - [ ] Auto-detect available hardware
  - [ ] Estimate memory requirements
  - [ ] Choose optimal backend based on data size
- [ ] Create unified backend interface
  - [ ] Ensure API consistency across backends
  - [ ] Implement fallback mechanisms
  - [ ] Add backend switching capability
  - [ ] Write integration tests
- [ ] Optimize data transfers
  - [ ] Minimize CPU-GPU transfers
  - [ ] Implement data prefetching
  - [ ] Use unified memory where available
  - [ ] Profile transfer overhead
- [ ] Implement chunking for large datasets
  - [ ] Calculate optimal chunk size
  - [ ] Handle chunk boundary conditions
  - [ ] Maintain statistical validity
  - [ ] Write tests with various chunk sizes

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
- [ ] Create permutation module structure
  - [ ] Implement src/gpu_randomise/core/permutation.py
  - [ ] Define permutation strategy interface
  - [ ] Create permutation generator base class
- [ ] Implement sign-flipping permutation
  - [ ] Create sign-flip strategy for paired designs
  - [ ] Generate permutation matrices
  - [ ] Validate permutation properties
  - [ ] Write unit tests
- [ ] Implement exchangeability blocks
  - [ ] Parse exchangeability file (.grp)
  - [ ] Handle block-wise permutations
  - [ ] Validate block constraints
  - [ ] Write tests with various block structures
- [ ] Create permutation scheduler
  - [ ] Manage permutation batching
  - [ ] Distribute across GPU cores
  - [ ] Track permutation progress
  - [ ] Implement checkpointing

### Week 10: Advanced Permutation Strategies
- [ ] Implement full enumeration
  - [ ] Detect when full enumeration is feasible
  - [ ] Generate all possible permutations
  - [ ] Handle memory constraints
  - [ ] Write tests for small samples
- [ ] Implement Monte Carlo sampling
  - [ ] Create random permutation generator
  - [ ] Ensure statistical validity
  - [ ] Set appropriate seed management
  - [ ] Validate distribution properties
- [ ] Implement variance smoothing
  - [ ] Add smoothing for t-statistics
  - [ ] Validate smoothing parameters
  - [ ] Compare with FSL implementation
  - [ ] Write unit tests
- [ ] Optimize permutation on GPU
  - [ ] Batch permutation operations
  - [ ] Minimize memory allocation
  - [ ] Parallelize across permutations
  - [ ] Profile and optimize

### Week 11: TFCE Implementation
- [ ] Research TFCE algorithm
  - [ ] Study Smith & Nichols (2009) paper
  - [ ] Examine FSL source code
  - [ ] Create implementation plan
- [ ] Implement basic TFCE
  - [ ] Create src/gpu_randomise/core/tfce.py
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

### Week 12: Multiple Comparison Corrections
- [ ] Implement FWER control
  - [ ] Create src/gpu_randomise/core/corrections.py
  - [ ] Implement max-statistic method
  - [ ] Calculate corrected p-values
  - [ ] Write unit tests
- [ ] Implement voxel-wise correction
  - [ ] Apply Bonferroni correction
  - [ ] Implement FDR correction
  - [ ] Validate correction methods
  - [ ] Write tests with synthetic data
- [ ] Implement cluster-based correction
  - [ ] Detect clusters above threshold
  - [ ] Calculate cluster statistics
  - [ ] Apply cluster-wise correction
  - [ ] Generate cluster tables
- [ ] Integrate corrections with permutation
  - [ ] Build null distributions
  - [ ] Calculate empirical p-values
  - [ ] Apply chosen correction method
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
- Total tasks: 57
- Completed: 57 (Week 1 + Week 2 + NIfTI implementation from Week 3)
- In Progress: 0
- Blocked: 0
- **Progress: 100%**

### Phase 2: GPU Acceleration
- Total tasks: 36
- Completed: 0
- In Progress: 0
- Blocked: 0
- **Progress: 0%**

### Phase 3: Statistical Features
- Total tasks: 44
- Completed: 0
- In Progress: 0
- Blocked: 0
- **Progress: 0%**

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

### Week 3 Progress (Data I/O Layer)
- NIfTI handling: **COMPLETE** (10/10 subtasks)
- Design matrix loader: **COMPLETE** (10/10 subtasks)
- Contrast file loader: **PENDING** (0/4 subtasks) 
- Output writer module: **PENDING** (0/4 subtasks)
- **Week 3 Progress: 83%** (20/24 subtasks complete)

### Overall Project
- **Total tasks: 221**
- **Completed: 67 (30%)**
- **Remaining: 154**

---

## Notes
- This task list is comprehensive but not exhaustive
- Add new tasks as they are discovered during development
- Some tasks may be split or combined based on actual complexity
- Priority may shift based on user feedback and requirements
- Update progress percentages after each work session

---

*Last Updated: 2025-08-27 (Refresh - NIfTI I/O Complete)*
*Next Review: 2025-09-03*
