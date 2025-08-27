# Project Requirements Document (PRD)
## GPU-Accelerated Permutation Testing for fMRI Data

### 1. Executive Summary

#### 1.1 Project Overview
Development of a high-performance Python application for GPU-accelerated permutation testing on fMRI neuroimaging data, inspired by FSL's randomise program but leveraging modern GPU architectures (CUDA and Apple Metal Performance Shaders) for substantial performance improvements.

#### 1.2 Business Case
- **Current Challenge**: FSL randomise is CPU-bound and can take hours to days for large datasets with many permutations
- **Opportunity**: Modern GPUs can parallelize permutation testing, potentially achieving 10-100x speedup
- **Target Users**: Neuroimaging researchers, computational neuroscientists, and clinical research teams

#### 1.3 Success Criteria
- Achieve minimum 10x performance improvement over CPU-based FSL randomise
- Maintain statistical accuracy within 0.001% of reference implementation
- Support both NVIDIA (CUDA) and Apple Silicon (MPS) GPUs
- Provide comprehensive test coverage (>90%)

### 2. Functional Requirements

The requirements for this package represent a subset of the full capability of the FSL randomise tool.
Additional features may be added in the future.

#### 2.1 Core Statistical Functions

##### 2.1.1 Permutation Testing Engine
- **General Linear Model (GLM) support**
  - Support for design matrices and contrast matrices
  - Handle multiple contrasts simultaneously
  - Support for F-tests and t-tests
  
- **Permutation Strategies**
  - Sign-flipping for paired/repeated measures designs
  - Full permutation enumeration (for small sample sizes)
  - Monte Carlo permutation sampling (for large sample sizes)
  - Block permutation for exchangeability blocks

- **Variance smoothing**
  - Variance smoothing for t-stats

##### 2.1.2 Statistical Corrections
- **Multiple Comparison Corrections**
  - Family-wise error rate (FWER) control
  - Threshold-free cluster enhancement (TFCE)
  - Cluster-based thresholding
  - Voxel-wise correction

##### 2.1.3 Test Statistics
- T-statistic computation
- F-statistic computation

#### 2.2 Data Input/Output

##### 2.2.1 Supported Input Formats
- **Neuroimaging Image Formats**
  - NIfTI (.nii, .nii.gz)

- **Design Files**
  - Text-based design matrices (CSV, TSV)
  - Text-based contrast files (CSV, TSV)

##### 2.2.2 Output Capabilities
- Output maps should use same format as input maps
- T/F-statistic maps
- Uncorrected p-value maps
- Voxel-based familywise error (FWE) corrected maps
- Cluster-based FWE corrected maps
- Cluster tables
- TFCE-corrected maps


#### 2.3 GPU Acceleration Features

##### 2.3.1 Apple Metal Performance Shaders (MPS) Support
- MPS backend for M1/M2/M3 Mac systems
- Automatic fallback to CPU for unsupported operations
- Memory optimization for Apple's unified memory architecture

##### 2.3.2 CUDA Support (to be added after MPS)
- CUDA compute capability 6.0+ support
- Dynamic kernel optimization based on data dimensions
- Efficient memory management with unified memory where applicable
- CUDA streams for overlapping computation and data transfer

##### 2.3.3 Automatic Backend Selection
- Runtime detection of available GPU resources
- Intelligent backend selection based on:
  - Data size
  - Available GPU memory
  - Operation complexity
  - User preferences

### 3. Non-Functional Requirements

#### 3.1 Performance Requirements
- **Throughput**: Process 100,000+ voxels with 10,000 permutations in under 1 minute on modern GPU
- **Memory Efficiency**: Handle datasets up to available GPU memory (typically 8-48GB)
- **Latency**: Interactive response (<100ms) for parameter adjustment in GUI mode

#### 3.2 Code Quality & Architecture

##### 3.2.1 Architecture Principles
- **Modular Design**
  - Clear separation of concerns (statistics, I/O, GPU kernels)
  - Dependency injection for testability

- **Design Patterns**
  - Strategy pattern for backend selection
  - Factory pattern for data loader creation
  - Observer pattern for progress reporting
  - Command pattern for operation queuing

##### 3.2.2 Code Standards
- PEP 8 compliance for Python code
- Type hints for all public APIs
- Comprehensive docstrings (NumPy style)
- Preferred maximum function length of 50 lines

#### 3.3 Testing Requirements

##### 3.3.1 Test Coverage
- Unit test coverage: >90%
- Integration test coverage: >80%
- GPU kernel testing with known inputs/outputs
- Statistical validation against FSL randomise outputs

##### 3.3.2 Test Framework
- pytest for test orchestration
- pytest-benchmark for performance regression testing
- pytest-cov for coverage reporting
- Mock GPU backends for CI/CD testing

##### 3.3.3 Test Categories
- **Unit Tests**: Individual function validation
- **Integration Tests**: Module interaction testing
- **Statistical Tests**: Accuracy validation against known results
- **Performance Tests**: Regression testing for speed/memory
- **Stress Tests**: Large dataset and edge case handling

#### 3.4 Documentation Requirements
- **User Documentation**
  - Installation guide for different platforms
  - Quick start tutorial
  - API reference documentation
  - Statistical methodology documentation
  - Migration guide from FSL randomise

- **Developer Documentation**
  - Architecture overview
  - Contributing guidelines
  - GPU kernel development guide
  - Plugin development guide

### 4. Technical Architecture

#### 4.1 System Architecture

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
│  │  NIfTI   │ │   CIFTI   │ │   Design  │      │
│  │  Loader  │ │   Loader  │ │   Loader  │      │
│  └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────┘
```

#### 4.2 Key Components

##### 4.2.1 Core Engine
- **Orchestrator**: Coordinates workflow execution
- **Statistics Module**: Implements GLM and test statistics
- **Permutation Engine**: Manages permutation generation and application
- **Correction Module**: Implements multiple comparison corrections

##### 4.2.2 Backend Abstraction
- **Backend Interface**: Abstract base class for compute backends
- **MPS Backend**: PyTorch MPS or Metal-cpp implementation
- **CUDA Backend**: CuPy/PyTorch-based CUDA implementation
- **CPU Backend**: NumPy/Numba fallback implementation

##### 4.2.3 Data Management
- **Data Loaders**: Format-specific data reading
- **Memory Manager**: GPU memory allocation and caching
- **Data Transformer**: Preprocessing and format conversion

#### 4.3 Technology Stack

##### 4.3.1 Core Dependencies
- **Python**: 3.12+ (for modern type hints and features)
- **NumPy**: Array operations and CPU backend
- **SciPy**: Statistical functions and distributions
- **Nibabel**: Neuroimaging data I/O
- **PyTorch**: GPU acceleration framework (CUDA and MPS)

##### 4.3.2 Development Dependencies
- **uv**: Package management
- **pytest**: Testing framework
- **ruff**: Code formatting and linting
- **mypy**: Static type checking
- **sphinx**: Documentation generation

### 5. Development Methodology

#### 5.1 Test-Driven Development (TDD) Approach
1. **Red Phase**: Write failing tests for new features
2. **Green Phase**: Implement minimum code to pass tests
3. **Refactor Phase**: Optimize and clean up code

#### 5.2 Development Workflow
1. **Feature Planning**: Create detailed specifications with test cases
2. **Research**: Examine existing FSL code and generate pseudocode for relevant functions
3. **Test Implementation**: Write comprehensive test suite
4. **Implementation**: Develop feature following TDD
5. **Code Review**: Peer review with focus on performance and correctness
6. **Integration Testing**: Validate with real fMRI datasets
7. **Performance Profiling**: Optimize bottlenecks
8. **Documentation**: Update user and API documentation

#### 5.3 Continuous Integration/Deployment
- **CI Pipeline**
  - Automated testing on push
  - Code quality checks (linting, formatting, type checking)
  - Performance regression testing
  - Documentation building

### 6. Validation & Benchmarking

#### 6.1 Statistical Validation
- Compare outputs with FSL randomise on standard datasets
- Synthetic data testing with known ground truth
- Edge case testing (small samples, extreme values)

#### 6.2 Performance Benchmarking
- **Benchmark Datasets**
  - Small: 1000 voxels, 20 subjects, 1000 permutations
  - Large: 250,000 voxels, 100 subjects, 10000 permutations

- **Metrics**
  - Wall-clock time
  - GPU utilization
  - Memory usage (GPU and system)
  - Speedup vs CPU implementation
  - Scaling efficiency (multi-GPU)

### 7. Project Timeline & Milestones

#### Phase 1: Foundation (Weeks 1-4)
- [ ] Project setup and CI pipeline
- [ ] Core architecture implementation
- [ ] Basic data I/O for NIfTI files
- [ ] CPU backend with basic GLM

#### Phase 2: GPU Acceleration (Weeks 5-8)
- [ ] MPS backend implementation
- [ ] CUDA backend implementation
- [ ] Backend abstraction layer
- [ ] GPU memory management

#### Phase 3: Statistical Features (Weeks 9-12)
- [ ] Permutation strategies implementation
- [ ] TFCE implementation
- [ ] Multiple comparison corrections
- [ ] Statistical validation suite

#### Phase 4: Optimization & Polish (Weeks 13-16)
- [ ] Performance optimization
- [ ] Comprehensive documentation
- [ ] Example notebooks and tutorials

#### Phase 5: Release Preparation (Weeks 17-18)
- [ ] Final testing and validation
- [ ] Package distribution setup
- [ ] Release documentation
- [ ] Community feedback integration

### 8. Risk Management

#### 8.1 Technical Risks
- **Risk**: GPU memory limitations for large datasets
  - **Mitigation**: Implement chunking and out-of-core processing
  
- **Risk**: Numerical precision differences between GPU and CPU
  - **Mitigation**: Extensive validation testing, mixed precision options

- **Risk**: Platform-specific GPU issues
  - **Mitigation**: Comprehensive testing matrix, fallback mechanisms

#### 8.2 Project Risks
- **Risk**: Scope creep from additional statistical methods
  - **Mitigation**: Clear phase boundaries

- **Risk**: Performance targets not met
  - **Mitigation**: Early profiling, alternative algorithmic approaches

### 9. Success Metrics

#### 9.1 Quantitative Metrics
- Performance improvement: >10x speedup vs FSL randomise
- Test coverage: >90% for unit tests
- Statistical accuracy: <0.001% deviation from reference

#### 9.2 Qualitative Metrics
- Documentation completeness and clarity
- Ease of migration from FSL randomise

### 10. Appendices

#### A. Glossary
- **fMRI**: Functional Magnetic Resonance Imaging
- **GLM**: General Linear Model
- **TFCE**: Threshold-Free Cluster Enhancement
- **FWER**: Family-Wise Error Rate
- **MPS**: Metal Performance Shaders

#### B. Reference Materials
- FSL Randomise Documentation: https://fsl.fmrib.ox.ac.uk/fsl/docs/#/statistics/randomise
- Winkler et al. (2014) - Permutation inference for the GLM
- Smith & Nichols (2009) - Threshold-free cluster enhancement
- Freedman & Lane (1983) - Permutation tests for linear models

#### C. Example Use Cases (with relevant randomise calls)

1. One-sample t-test on fMRI data: randomise -i <input> -o <output> -1 -T -c 3.0 -v 10 
2. One-sample t-test on fMRI data with exchangeability blocks: randomise -i <input> -o <output> -1 -T -c 3.0 -v 10 -e <design.grp> --permuteBlocks
3. Multiple regression with t contrasts on fMRI data: randomise -i <input> -o <output> -d <design matrix file> -t <t contrasts file> -T -c 3.0 -v 10 -e <design.grp> --permuteBlocks