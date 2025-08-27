Quick Start Guide
=================

This guide will get you started with AccelPerm quickly.

Basic Usage
-----------

AccelPerm is designed as a drop-in replacement for FSL randomise.
Most FSL randomise commands work directly with AccelPerm.

One-Sample T-Test
~~~~~~~~~~~~~~~~~

Test if the mean is significantly different from zero::

    accelperm -i input.nii -o output -1 -T -c 3.0 -v 1000

* ``-i input.nii``: Input 4D image
* ``-o output``: Output prefix
* ``-1``: One-sample t-test
* ``-T``: Output t-statistic maps
* ``-c 3.0``: Cluster threshold
* ``-v 1000``: Number of permutations

Two-Sample T-Test
~~~~~~~~~~~~~~~~~

Compare two groups using a design matrix::

    accelperm -i input.nii -o output -d design.mat -t design.con -T -c 3.0 -v 1000

The design matrix (``design.mat``) should have one column per group::

    1 0
    1 0
    1 0
    0 1
    0 1
    0 1

The contrast file (``design.con``) specifies the comparison::

    1 -1

Multiple Regression
~~~~~~~~~~~~~~~~~~~

Test for effects of continuous variables::

    accelperm -i input.nii -o output -d design.mat -t design.con -T -v 1000

Example design matrix with age and group::

    25 1 0  # Subject 1: age 25, group 1
    30 1 0  # Subject 2: age 30, group 1
    28 0 1  # Subject 3: age 28, group 2
    35 0 1  # Subject 4: age 35, group 2

Contrast to test age effect::

    1 0 0

TFCE (Recommended)
~~~~~~~~~~~~~~~~~~

Use Threshold-Free Cluster Enhancement for better sensitivity::

    accelperm -i input.nii -o output -1 -T --tfce -v 5000

TFCE doesn't require setting a cluster threshold and is generally more powerful.

Advanced Options
----------------

Exchangeability Blocks
~~~~~~~~~~~~~~~~~~~~~~

For repeated measures or family data, specify exchangeability constraints::

    accelperm -i input.nii -o output -1 -T -v 1000 -e design.grp

The ``.grp`` file specifies which subjects can be permuted together::

    1
    1
    2
    2
    3
    3

Subjects with the same number form an exchangeability block.

Custom Thresholds
~~~~~~~~~~~~~~~~~

Set different cluster-forming thresholds::

    accelperm -i input.nii -o output -1 -T -c 2.3 -v 1000  # t > 2.3
    accelperm -i input.nii -o output -1 -F -c 4.0 -v 1000  # F > 4.0

Multiple Correction Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``-c``: Cluster-based correction
* ``--tfce``: Threshold-Free Cluster Enhancement
* ``--fdr``: False Discovery Rate
* ``--bonf``: Bonferroni correction

GPU Backend Selection
~~~~~~~~~~~~~~~~~~~~~

AccelPerm automatically selects the best available backend, but you can force selection::

    accelperm --backend cuda -i input.nii -o output -1 -T -v 1000
    accelperm --backend mps -i input.nii -o output -1 -T -v 1000
    accelperm --backend cpu -i input.nii -o output -1 -T -v 1000

Performance Tips
----------------

1. **Use more permutations for final results**: 10,000+ permutations for publication
2. **Start with fewer permutations for testing**: 1,000 permutations for development
3. **Use TFCE when possible**: Generally more powerful than cluster correction
4. **Monitor GPU memory**: Large datasets may need chunking
5. **Use multiple cores**: AccelPerm automatically uses all available CPU cores as fallback

Common Workflows
----------------

Quality Assurance
~~~~~~~~~~~~~~~~~

Always start with a small number of permutations to verify the analysis::

    accelperm -i input.nii -o test_output -1 -T -v 100

Check the output files make sense before running the full analysis.

Comparing Methods
~~~~~~~~~~~~~~~~~

Compare different correction methods::

    # Cluster correction
    accelperm -i input.nii -o cluster_output -1 -T -c 3.0 -v 5000

    # TFCE
    accelperm -i input.nii -o tfce_output -1 -T --tfce -v 5000

    # FDR
    accelperm -i input.nii -o fdr_output -1 -T --fdr -v 5000

Expected Outputs
----------------

AccelPerm produces the same output files as FSL randomise:

* ``output_tstat1.nii.gz``: T-statistic map
* ``output_tfce_tstat1.nii.gz``: TFCE-enhanced t-statistics
* ``output_tstat1_p.nii.gz``: Uncorrected p-values
* ``output_tfce_corrp_tstat1.nii.gz``: TFCE-corrected p-values
* ``output_clusters.txt``: Cluster information table

Next Steps
----------

* Read the :doc:`api` documentation for detailed parameter descriptions
* Check out :doc:`examples` for more complex analyses
* See :doc:`contributing` if you want to contribute to development
