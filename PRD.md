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
  - Support for custom permutation schemes

##### 2.1.2 Statistical Corrections
- **Multiple Comparison Corrections**
  - Family-wise error rate (FWER) control
  - Threshold-free cluster enhancement (TFCE)
  - Cluster-based thresholding
  - Voxel-wise correction
  - False discovery rate (FDR) options

##### 2.1.3 Test Statistics
- T-statistic computation
- F-statistic computation
- Pseudo-t statistic for heteroscedastic data
- Support for custom test statistics via plugin architecture

#### 2.2 Data Input/Output

##### 2.2.1 Supported Input Formats
- **Neuroimaging Formats**
  - NIfTI (.nii, .nii.gz)
  - CIFTI-2 (.dscalar.nii, .dtseries.nii)
  - FSL format files (.nii.gz with .bvec/.bval for DTI)
  - Surface data (GIFTI format)

- **Design Files**
  - FSL format design matrices (.mat)
  - FSL format contrast files (.con)
  - Text-based design matrices (CSV, TSV)
  - NumPy array format (.npy, .npz)

##### 2.2.2 Output Capabilities
- Corrected and uncorrected p-value maps
- T/F-statistic maps
- TFCE output maps
- Cluster maps and tables
- Null distribution histograms
- Maximum statistic distributions
- Report generation (HTML, PDF)

#### 2.3 GPU Acceleration Features

##### 2.3.1 CUDA Support
- CUDA compute capability 6.0+ support
- Multi-GPU support for large-scale analyses
- Dynamic kernel optimization based on data dimensions
- Efficient memory management with unified memory where applicable
- CUDA streams for overlapping computation and data transfer

##### 2.3.2 Apple Metal Performance Shaders (MPS) Support
- MPS backend for M1/M2/M3 Mac systems
- Automatic fallback to CPU for unsupported operations
- Memory optimization for Apple's unified memory architecture

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
- **Scalability**: Linear scaling with number of GPUs for multi-GPU systems
- **Latency**: Interactive response (<100ms) for parameter adjustment in GUI mode

#### 3.2 Code Quality & Architecture

##### 3.2.1 Architecture Principles
- **Modular Design**
  - Clear separation of concerns (statistics, I/O, GPU kernels)
  - Plugin architecture for custom statistics and corrections
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
- Maximum cyclomatic complexity of 10
- Maximum function length of 50 lines

#### 3.3 Testing Requirements

##### 3.3.1 Test Coverage
- Unit test coverage: >90%
- Integration test coverage: >80%
- GPU kernel testing with known inputs/outputs
- Statistical validation against FSL randomise outputs

##### 3.3.2 Test Framework
- pytest for test orchestration
- hypothesis for property-based testing
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
│  │   CUDA   │ │    MPS    │ │    CPU   │      │
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
- **CUDA Backend**: CuPy/PyTorch-based CUDA implementation
- **MPS Backend**: PyTorch MPS or Metal-cpp implementation
- **CPU Backend**: NumPy/Numba fallback implementation

##### 4.2.3 Data Management
- **Data Loaders**: Format-specific data reading
- **Memory Manager**: GPU memory allocation and caching
- **Data Transformer**: Preprocessing and format conversion

#### 4.3 Technology Stack

##### 4.3.1 Core Dependencies
- **Python**: 3.9+ (for modern type hints and features)
- **NumPy**: Array operations and CPU backend
- **SciPy**: Statistical functions and distributions
- **Nibabel**: Neuroimaging data I/O
- **PyTorch**: GPU acceleration framework (CUDA and MPS)
- **CuPy** (optional): Alternative CUDA backend

##### 4.3.2 Development Dependencies
- **pytest**: Testing framework
- **black**: Code formatting
- **mypy**: Static type checking
- **ruff**: Fast Python linter
- **sphinx**: Documentation generation
- **pre-commit**: Git hooks for code quality

##### 4.3.3 Optional Dependencies
- **Dask**: Distributed computing support
- **Ray**: Alternative distributed computing
- **Matplotlib/Plotly**: Visualization
- **Streamlit/Gradio**: Web-based GUI

### 5. Development Methodology

#### 5.1 Test-Driven Development (TDD) Approach
1. **Red Phase**: Write failing tests for new features
2. **Green Phase**: Implement minimum code to pass tests
3. **Refactor Phase**: Optimize and clean up code

#### 5.2 Development Workflow
1. **Feature Planning**: Create detailed specifications with test cases
2. **Test Implementation**: Write comprehensive test suite
3. **Implementation**: Develop feature following TDD
4. **Code Review**: Peer review with focus on performance and correctness
5. **Integration Testing**: Validate with real fMRI datasets
6. **Performance Profiling**: Optimize bottlenecks
7. **Documentation**: Update user and API documentation

#### 5.3 Continuous Integration/Deployment
- **CI Pipeline**
  - Automated testing on push
  - Code quality checks (linting, formatting, type checking)
  - Performance regression testing
  - Documentation building

- **CD Pipeline**
  - Automated package building
  - PyPI deployment for releases
  - Docker image creation
  - Conda package building

### 6. Validation & Benchmarking

#### 6.1 Statistical Validation
- Compare outputs with FSL randomise on standard datasets
- Validate against published results from neuroimaging studies
- Synthetic data testing with known ground truth
- Edge case testing (small samples, extreme values)

#### 6.2 Performance Benchmarking
- **Benchmark Datasets**
  - Small: 1000 voxels, 20 subjects, 1000 permutations
  - Medium: 100,000 voxels, 100 subjects, 5000 permutations
  - Large: 1,000,000 voxels, 500 subjects, 10000 permutations

- **Metrics**
  - Wall-clock time
  - GPU utilization
  - Memory usage (GPU and system)
  - Speedup vs CPU implementation
  - Scaling efficiency (multi-GPU)

### 7. Project Timeline & Milestones

#### Phase 1: Foundation (Weeks 1-4)
- [ ] Project setup and CI/CD pipeline
- [ ] Core architecture implementation
- [ recursion ] Basic data I/O for NIfTI files
- [ ] CPU backend with basic GLM

#### Phase 2: GPU Acceleration (Weeks 5-8)
- [ ] CUDA backend implementation
- [ ] MPS backend implementation
- [ ] Backend abstraction layer
- [ ] GPU memory management

#### Phase 3: Statistical Features (Weeks 9-12)
- [ ] Permutation strategies implementation
- [ ] TFCE implementation
- [ ] Multiple comparison corrections
- [ ] Statistical validation suite

#### Phase 4: Optimization & Polish (Weeks 13-16)
- [ ] Performance optimization
- [ ] Multi-GPU support
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
  - **Mitigation**: Clear phase boundaries, plugin architecture for extensions

- **Risk**: Performance targets not met
  - **Mitigation**: Early profiling, alternative algorithmic approaches

### 9. Success Metrics

#### 9.1 Quantitative Metrics
- Performance improvement: >10x speedup vs FSL randomise
- Test coverage: >90% for unit tests
- Statistical accuracy: <0.001% deviation from reference
- Memory efficiency: <2x memory overhead vs theoretical minimum
- User adoption: 100+ users within 6 months

#### 9.2 Qualitative Metrics
- Code maintainability score: A rating (via CodeClimate or similar)
- Documentation completeness and clarity
- Community engagement and contributions
- Ease of migration from FSL randomise

### 10. Appendices

#### A. Glossary
- **fMRI**: Functional Magnetic Resonance Imaging
- **GLM**: General Linear Model
- **TFCE**: Threshold-Free Cluster Enhancement
- **FWER**: Family-Wise Error Rate
- **FDR**: False Discovery Rate
- **MPS**: Metal Performance Shaders

#### B. Reference Materials
- FSL Randomise Documentation
- Winkler et al. (2014) - Permutation inference for the GLM
- Smith & Nichols (2009) - Threshold-free cluster enhancement
- Freedman & Lane (1983) - Permutation tests for linear models

#### C. Example Use Cases
1. Whole-brain voxel-wise analysis of task fMRI data
2. Surface-based analysis of cortical thickness
3. Connectome-wide association studies
4. DTI-based tract analysis
5. Longitudinal neuroimaging studies