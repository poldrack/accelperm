Examples
========

This section provides detailed examples of using AccelPerm for common neuroimaging analyses.

Example 1: Basic One-Sample T-Test
----------------------------------

This example tests whether brain activation is significantly greater than zero.

**Data**: 20 subjects, single contrast from first-level analysis

**Command**::

    accelperm -i group_cope.nii.gz -o onesample -1 -T -c 2.3 -v 5000

**Expected Output**:

* ``onesample_tstat1.nii.gz``: T-statistic map
* ``onesample_tstat1_p.nii.gz``: Uncorrected p-values
* ``onesample_clustere_corrp_tstat1.nii.gz``: Cluster-corrected p-values
* ``onesample_clusters.txt``: Cluster information table

**Interpretation**: Voxels with p < 0.05 in the corrected p-value map show
significant activation after correction for multiple comparisons.

Example 2: Two-Sample T-Test
----------------------------

Compare brain activation between two groups (e.g., patients vs controls).

**Data**: 15 patients, 15 controls

**Design Matrix** (``design.mat``)::

    1 0  # Patient 1
    1 0  # Patient 2
    ...
    1 0  # Patient 15
    0 1  # Control 1
    0 1  # Control 2
    ...
    0 1  # Control 15

**Contrast** (``design.con``)::

    1 -1  # Patients > Controls

**Command**::

    accelperm -i all_subjects.nii.gz -o twosample -d design.mat -t design.con -T -c 2.3 -v 10000

Example 3: TFCE Analysis
------------------------

Use Threshold-Free Cluster Enhancement for increased sensitivity.

**Command**::

    accelperm -i group_cope.nii.gz -o tfce_analysis -1 --tfce -v 10000

**Key Advantages of TFCE**:

* No arbitrary threshold selection
* Better sensitivity than cluster correction
* Preserves spatial information
* Works well with smooth and focal activations

**Output Interpretation**: The ``tfce_corrp_tstat1.nii.gz`` file contains
TFCE-corrected p-values. Values < 0.05 indicate significant effects.

Example 4: Multiple Regression
------------------------------

Test effects of age while controlling for group differences.

**Design Matrix** (``design.mat``)::

    # Age  Group1  Group2
    25    1       0      # Young subject, Group 1
    30    1       0      # Old subject, Group 1
    28    0       1      # Young subject, Group 2
    35    0       1      # Old subject, Group 2
    ...

**Contrast for Age Effect** (``age.con``)::

    1 0 0  # Age effect

**Contrast for Group Difference** (``group.con``)::

    0 1 -1  # Group 1 > Group 2

**Commands**::

    # Test age effect
    accelperm -i data.nii.gz -o age_effect -d design.mat -t age.con --tfce -v 10000

    # Test group difference
    accelperm -i data.nii.gz -o group_diff -d design.mat -t group.con --tfce -v 10000

Example 5: Repeated Measures
----------------------------

Analyze longitudinal data with proper exchangeability constraints.

**Data**: 20 subjects scanned at 2 timepoints

**Design Matrix** (``design.mat``)::

    1 0  # Subject 1, Time 1
    0 1  # Subject 1, Time 2
    1 0  # Subject 2, Time 1
    0 1  # Subject 2, Time 2
    ...

**Exchangeability Blocks** (``design.grp``)::

    1
    1
    2
    2
    3
    3
    ...

**Contrast** (``time.con``)::

    -1 1  # Time 2 > Time 1

**Command**::

    accelperm -i longitudinal.nii.gz -o timepoint -d design.mat -t time.con -e design.grp --tfce -v 10000

Example 6: F-Test for Overall Effect
------------------------------------

Test for any effect across multiple conditions using F-statistics.

**Design Matrix** (``design.mat``)::

    # Cond1  Cond2  Cond3
    1      0      0     # Condition 1
    0      1      0     # Condition 2
    0      0      1     # Condition 3
    ...

**F-Contrast** (``ftest.fts``)::

    1 0 0
    0 1 0
    0 0 1

**Command**::

    accelperm -i conditions.nii.gz -o ftest -d design.mat -f ftest.fts -F --tfce -v 10000

Example 7: Performance Comparison
---------------------------------

Compare AccelPerm performance against FSL randomise.

**FSL randomise**::

    randomise -i data.nii.gz -o fsl_output -1 -T -c 2.3 -v 5000

**AccelPerm**::

    accelperm -i data.nii.gz -o accel_output -1 -T -c 2.3 -v 5000

**Validation Script** (Python)::

    import nibabel as nib
    import numpy as np

    # Load results
    fsl_tstat = nib.load('fsl_output_tstat1.nii.gz').get_fdata()
    accel_tstat = nib.load('accel_output_tstat1.nii.gz').get_fdata()

    # Calculate difference
    diff = np.abs(fsl_tstat - accel_tstat)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Maximum difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    print(f"Relative error: {mean_diff / np.mean(np.abs(fsl_tstat)) * 100:.6f}%")

Expected output: < 0.001% relative error

Example 8: Large Dataset Processing
-----------------------------------

Process large datasets with automatic memory management.

**Command**::

    accelperm -i large_dataset.nii.gz -o large_output -1 --tfce -v 10000 --verbose

**Monitoring**:

* Use ``--verbose`` for detailed progress information
* AccelPerm automatically chunks data if GPU memory is insufficient
* Monitor GPU usage: ``nvidia-smi -l 1`` (NVIDIA) or Activity Monitor (macOS)

**Performance Tips**:

* Start with fewer permutations (1000) for testing
* Use TFCE instead of cluster correction when possible
* Consider using multiple GPUs for very large datasets

Example 9: Custom Configuration
-------------------------------

Use configuration files for complex analyses.

**Configuration File** (``config.toml``)::

    [analysis]
    input = "data.nii.gz"
    output_prefix = "custom_analysis"
    design_matrix = "design.mat"
    contrasts = ["contrast1.con", "contrast2.con"]

    [permutation]
    n_permutations = 10000
    strategy = "sign_flipping"

    [correction]
    method = "tfce"
    tfce_height = 2.0
    tfce_extent = 0.5

    [backend]
    preferred = "cuda"
    fallback = "cpu"
    chunk_size = 1000

**Command**::

    accelperm --config config.toml

Example 10: Batch Processing
----------------------------

Process multiple datasets efficiently.

**Batch Script** (``process_batch.sh``)::

    #!/bin/bash

    for subject in sub-01 sub-02 sub-03; do
        echo "Processing $subject..."
        accelperm -i ${subject}_cope.nii.gz \
                 -o ${subject}_results \
                 -1 --tfce -v 5000
    done

    echo "All subjects processed!"

**Python Batch Processing**::

    import subprocess
    from pathlib import Path

    data_dir = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    for cope_file in data_dir.glob("*_cope.nii.gz"):
        subject_id = cope_file.stem.replace("_cope", "")
        output_prefix = results_dir / subject_id

        cmd = [
            "accelperm",
            "-i", str(cope_file),
            "-o", str(output_prefix),
            "-1", "--tfce", "-v", "5000"
        ]

        subprocess.run(cmd, check=True)
        print(f"Completed: {subject_id}")

Best Practices
--------------

1. **Always validate with FSL**: Compare results on a subset of data
2. **Use appropriate permutation counts**:

   * Development: 1,000 permutations
   * Production: 5,000-10,000 permutations
   * Publication: 10,000+ permutations

3. **Choose the right correction method**:

   * TFCE: Best for most analyses
   * Cluster: When you need cluster extent information
   * FDR: For very conservative corrections

4. **Monitor resources**: Check GPU memory and CPU usage
5. **Save intermediate results**: Use ``--save-permutations`` for debugging
6. **Document your analysis**: Keep analysis scripts and parameters
