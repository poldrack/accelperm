# Development scratchpad

- Use this file to keep notes on ongoing development work.
- When the work is completed, clean it out from this file, so that it only reflects ongoing work.

## Project State Summary (2025-08-28)

### Current Status
**Branch**: dev/week7-backend-optimization
**Phase**: Phase 3 - Statistical Features
**Overall Progress**: 170/262 tasks completed (64.9%)
**Current Week**: Week 12 - TFCE Implementation (Ready to Start)

### Recently Completed

#### Week 11: Multiple Comparison Corrections (100% Complete) ✅
- ✅ **Complete corrections module** - `src/accelperm/core/corrections.py`
  - CorrectionResult dataclass for standardized result handling
  - CorrectionMethod ABC with validation and error handling
  - BonferroniCorrection: Conservative FWER control with alpha/n adjustment
  - FDRCorrection: Benjamini-Hochberg procedure with conservative mode option
  - FWERCorrection: Max-statistic method using permutation null distributions
  - ClusterCorrection: Spatial extent and mass correction with 3D connectivity options

- ✅ **CLI Integration** - Enhanced command-line interface
  - New correction parameters: --correction, --alpha, --n-perm
  - Support for correction methods: none, bonferroni, fdr, fwer, cluster
  - Automatic fallback warnings for methods requiring permutation testing
  - Enhanced output: corrected p-values, significance masks, detailed summaries

- ✅ **Comprehensive Testing** - `tests/unit/test_corrections.py`
  - 27 tests total, 100% pass rate, 87.44% code coverage
  - TDD methodology: Proper RED-GREEN-REFACTOR cycle completed
  - Integration tests covering method comparisons and realistic workflows
  - Validation tests covering error handling and edge cases

- ✅ **Architecture Features**
  - Abstract base class pattern for extensible correction methods
  - Strategy pattern for different correction approaches
  - Input validation for p-values and alpha parameters
  - Statistical accuracy: proper FDR step-up, Bonferroni adjustment
  - Cluster analysis: 3D connectivity (6/18/26), extent vs mass correction
  - FSL randomise compatibility design patterns

### Current Git Status (Post Week 11)
- Latest commit: e2a3514 "feat: complete Week 11 Multiple Comparison Corrections"
- Working directory: Clean (some minor linting changes not committed)
- New files successfully integrated:
  - `src/accelperm/core/corrections.py` (complete implementation)
  - `tests/unit/test_corrections.py` (comprehensive test suite)
  - Enhanced `src/accelperm/cli.py` (correction parameters and output)

### Test Status Update
- **Total**: 264 tests (increased from 237)
- **Passing**: 264 tests (100% pass rate) 
- **Corrections Module**: 27 tests, 100% pass rate, 87.44% coverage
- **Overall Project Coverage**: Estimated 88%+ with corrections module

### Architecture Progress

#### Completed Architecture Components
1. **Data I/O Layer** ✅ - Complete NIfTI, design matrix, contrast file handling
2. **Backend Abstraction Layer** ✅ - CPU, MPS backends with factory selection
3. **Core Engine - Statistics** ✅ - GLM computation with multiple backends  
4. **Core Engine - Permutation** ✅ - Advanced permutation strategies
5. **Core Engine - Correction** ✅ - Multiple comparison corrections
6. **CLI Interface** ✅ - Full command-line interface with correction support

#### Next: Core Engine - TFCE
- Threshold-Free Cluster Enhancement implementation
- Critical for neuroimaging spatial statistics
- Final major statistical component needed

### Immediate Next Steps (Week 12: TFCE Implementation)

#### Research Phase
1. Study Smith & Nichols (2009) TFCE paper
2. Examine FSL randomise source code for TFCE implementation
3. Review PSEUDOCODE.md for FSL-compatible TFCE algorithms
4. Create detailed implementation plan

#### Implementation Phase (Following TDD)
1. **Create TFCE module structure**
   - `src/accelperm/core/tfce.py` 
   - Define TFCE class and interface
   - Write failing tests first (RED phase)

2. **Implement basic TFCE algorithm**
   - Threshold stepping with configurable dh parameter
   - Connected component labeling for cluster detection
   - Height (H) and extent (E) weighting calculations
   - Implement minimal code to pass tests (GREEN phase)

3. **Optimize and integrate**
   - GPU optimization for parallel threshold iterations
   - Memory optimization for large statistical maps
   - Integration with existing correction framework
   - Refactor for performance while maintaining tests (REFACTOR phase)

4. **Validation and testing**
   - Compare outputs with FSL randomise TFCE results
   - Test on known datasets for statistical accuracy
   - Comprehensive unit tests for edge cases
   - Performance benchmarking

#### Technical Considerations
- **TFCE Algorithm**: Threshold-free approach avoiding arbitrary thresholds
- **Mathematical Formula**: TFCE(h) = ∫[0→h] e(h')^E × h'^H dh'
- **Key Parameters**: 
  - dh: threshold step size (typically 0.1)
  - E: extent exponent (typically 0.5)  
  - H: height exponent (typically 2.0)
- **Performance**: Critical to optimize for large neuroimaging datasets

### Performance and Technical Notes

#### Current State
- MPS backend functional but not optimized for speed
- CPU-GPU data transfers still present performance bottleneck
- Chunking system prevents memory issues on large datasets
- Statistical accuracy prioritized over performance (as intended)

#### Architecture Decisions Made
1. **Correction Integration**: Seamlessly integrated with existing CLI and backend system
2. **Statistical Accuracy**: All corrections follow established neuroimaging standards
3. **Extensibility**: Abstract base class pattern allows easy addition of new correction methods
4. **User Experience**: Automatic fallback warnings guide users to appropriate methods

### Phase 3 Statistical Features Summary
- **Week 9: Permutation Engine Core** ✅ - 95% complete
- **Week 10: Advanced Permutation Strategies** ✅ - 100% complete  
- **Week 11: Multiple Comparison Corrections** ✅ - 100% complete
- **Week 12: TFCE Implementation** - Ready to start (0% complete)
- **Phase 3 Overall**: 75% complete

### Project Milestone Achievement
With Week 11 complete, the project now has:
- ✅ Complete foundation infrastructure
- ✅ GPU-accelerated backend system  
- ✅ Advanced permutation testing capabilities
- ✅ Comprehensive multiple comparison corrections
- 🎯 Ready for final statistical component (TFCE)

**Major Achievement**: The project now provides a nearly complete alternative to FSL randomise with modern GPU acceleration and comprehensive statistical corrections. Only TFCE implementation remains for Phase 3 completion.

### Known Issues
1. **Minor Linting Issues**: Some trailing whitespace and formatting issues remain
   - Pre-commit hooks cleaning up formatting automatically
   - Core functionality unaffected
   
2. **Performance Optimization Deferred**: Focus remains on statistical accuracy
   - MPS backend prioritizes correctness over speed
   - Performance optimization planned for Phase 4
   
3. **CUDA Backend**: Still not implemented (Week 6 deferred)
   - MPS backend covers Apple Silicon GPU acceleration
   - CUDA implementation can be added in optimization phase

### Success Metrics
- ✅ **Statistical Accuracy**: All corrections follow established standards
- ✅ **Test Coverage**: 87.44% coverage on corrections module
- ✅ **Code Quality**: Comprehensive TDD methodology followed
- ✅ **User Experience**: CLI integration with clear parameter options
- ✅ **Extensibility**: Clean architecture for future enhancements

### Ready for Week 12
The project is well-positioned for TFCE implementation with:
- Solid foundation of statistical components
- Proven TDD methodology
- Established architecture patterns
- Comprehensive testing framework
- CLI integration ready for TFCE parameters