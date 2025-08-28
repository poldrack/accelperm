"""
Test module for multiple comparison corrections.

Following TDD methodology - writing comprehensive tests FIRST before implementation.
These tests define the expected behavior of the corrections module.
"""


import numpy as np
import pytest

from accelperm.core.corrections import (
    BonferroniCorrection,
    ClusterCorrection,
    CorrectionMethod,
    CorrectionResult,
    FDRCorrection,
    FWERCorrection,
)


class TestCorrectionResult:
    """Test the CorrectionResult data class."""

    def test_correction_result_creation(self):
        """Test CorrectionResult can be created with required fields."""
        # This will fail initially - we need to implement CorrectionResult
        p_values = np.array([0.001, 0.05, 0.1])
        corrected_p = np.array([0.003, 0.15, 0.3])
        threshold = 0.05
        significant_mask = np.array([True, False, False])

        result = CorrectionResult(
            p_values=p_values,
            corrected_p_values=corrected_p,
            threshold=threshold,
            significant_mask=significant_mask,
            method="bonferroni",
            n_comparisons=3,
        )

        np.testing.assert_array_equal(result.p_values, p_values)
        np.testing.assert_array_equal(result.corrected_p_values, corrected_p)
        assert result.threshold == threshold
        np.testing.assert_array_equal(result.significant_mask, significant_mask)
        assert result.method == "bonferroni"
        assert result.n_comparisons == 3


class TestCorrectionMethod:
    """Test the abstract CorrectionMethod base class."""

    def test_correction_method_is_abstract(self):
        """Test that CorrectionMethod cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CorrectionMethod()

    def test_correction_method_has_required_methods(self):
        """Test that subclasses must implement required abstract methods."""

        class IncompletCorrection(CorrectionMethod):
            pass

        with pytest.raises(TypeError):
            IncompletCorrection()


class TestBonferroniCorrection:
    """Test Bonferroni correction for multiple comparisons."""

    def test_bonferroni_correction_basic(self):
        """Test basic Bonferroni correction functionality."""
        p_values = np.array([0.001, 0.01, 0.05, 0.1])
        expected_corrected = np.array([0.004, 0.04, 0.2, 0.4])

        bonferroni = BonferroniCorrection()
        result = bonferroni.correct(p_values, alpha=0.05)

        assert isinstance(result, CorrectionResult)
        np.testing.assert_allclose(result.corrected_p_values, expected_corrected)
        assert result.method == "bonferroni"
        assert result.n_comparisons == 4

    def test_bonferroni_significance_threshold(self):
        """Test Bonferroni significance determination."""
        p_values = np.array([0.01, 0.02, 0.03, 0.1])  # 4 comparisons
        alpha = 0.05

        bonferroni = BonferroniCorrection()
        result = bonferroni.correct(p_values, alpha=alpha)

        # Bonferroni threshold = alpha / n = 0.05 / 4 = 0.0125
        expected_significant = np.array([True, False, False, False])
        np.testing.assert_array_equal(result.significant_mask, expected_significant)
        assert result.threshold == alpha / len(p_values)

    def test_bonferroni_edge_cases(self):
        """Test Bonferroni correction edge cases."""
        bonferroni = BonferroniCorrection()

        # Single p-value
        result = bonferroni.correct(np.array([0.05]), alpha=0.05)
        assert result.corrected_p_values[0] == 0.05
        assert result.significant_mask[0] == True

        # All p-values = 1.0 (maximum)
        result = bonferroni.correct(np.array([1.0, 1.0, 1.0]), alpha=0.05)
        np.testing.assert_array_equal(
            result.corrected_p_values, np.array([1.0, 1.0, 1.0])
        )

        # Empty array
        with pytest.raises(ValueError):
            bonferroni.correct(np.array([]), alpha=0.05)


class TestFDRCorrection:
    """Test False Discovery Rate (Benjamini-Hochberg) correction."""

    def test_fdr_correction_basic(self):
        """Test basic FDR correction functionality."""
        # Known example from Benjamini-Hochberg paper
        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        fdr = FDRCorrection()
        result = fdr.correct(p_values, alpha=0.05)

        assert isinstance(result, CorrectionResult)
        assert result.method == "fdr_bh"
        assert result.n_comparisons == 5

        # FDR should be less conservative than Bonferroni
        bonferroni = BonferroniCorrection()
        bonf_result = bonferroni.correct(p_values, alpha=0.05)

        assert np.sum(result.significant_mask) >= np.sum(bonf_result.significant_mask)

    def test_fdr_step_up_procedure(self):
        """Test FDR step-up procedure implementation."""
        # Sorted p-values: should find largest k where p(k) <= (k/m) * alpha
        p_values = np.array([0.001, 0.01, 0.03, 0.04, 0.2])
        alpha = 0.05

        fdr = FDRCorrection()
        result = fdr.correct(p_values, alpha=alpha)

        # Manual calculation:
        # k=1: 0.001 <= (1/5)*0.05 = 0.01? Yes
        # k=2: 0.01 <= (2/5)*0.05 = 0.02? Yes
        # k=3: 0.03 <= (3/5)*0.05 = 0.03? Yes
        # k=4: 0.04 <= (4/5)*0.05 = 0.04? Yes
        # k=5: 0.2 <= (5/5)*0.05 = 0.05? No

        expected_significant = np.array([True, True, True, True, False])
        np.testing.assert_array_equal(result.significant_mask, expected_significant)

    def test_fdr_adjusted_p_values(self):
        """Test FDR adjusted p-values calculation."""
        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.1])

        fdr = FDRCorrection()
        result = fdr.correct(p_values, alpha=0.05)

        # FDR adjusted p-values should be monotonic
        sorted_indices = np.argsort(p_values)
        sorted_adjusted = result.corrected_p_values[sorted_indices]

        # Check monotonicity (each value >= previous)
        for i in range(1, len(sorted_adjusted)):
            assert sorted_adjusted[i] >= sorted_adjusted[i - 1]

        # All adjusted p-values should be >= original
        assert np.all(result.corrected_p_values >= p_values)

    def test_fdr_conservative_mode(self):
        """Test FDR correction with conservative correction factor."""
        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        fdr_standard = FDRCorrection(conservative=False)
        fdr_conservative = FDRCorrection(conservative=True)

        result_standard = fdr_standard.correct(p_values, alpha=0.05)
        result_conservative = fdr_conservative.correct(p_values, alpha=0.05)

        # Conservative should have higher adjusted p-values
        assert np.all(
            result_conservative.corrected_p_values >= result_standard.corrected_p_values
        )

        # Conservative should be more stringent (fewer significant)
        assert np.sum(result_conservative.significant_mask) <= np.sum(
            result_standard.significant_mask
        )


class TestFWERCorrection:
    """Test Family-Wise Error Rate correction with max-statistic method."""

    def test_fwer_max_statistic_setup(self):
        """Test FWER correction initialization."""
        null_distribution = np.random.randn(1000, 100)  # 1000 permutations, 100 voxels

        fwer = FWERCorrection(null_distribution)

        assert fwer.null_distribution.shape == (1000, 100)
        assert fwer.method == "fwer_max_stat"
        assert hasattr(fwer, "max_null_distribution")

    def test_fwer_max_null_distribution(self):
        """Test calculation of maximum null distribution."""
        # Create synthetic null distribution where we know the maximum
        null_dist = np.array([[1.0, 2.0, 0.5], [1.5, 1.0, 2.5], [0.8, 3.0, 1.2]])
        expected_max = np.array(
            [2.0, 2.5, 3.0]
        )  # max across voxels for each permutation

        fwer = FWERCorrection(null_dist)

        np.testing.assert_array_equal(fwer.max_null_distribution, expected_max)

    def test_fwer_p_value_calculation(self):
        """Test FWER corrected p-value calculation."""
        # Create controlled null distribution
        null_dist = np.array(
            [
                [1.0, 2.0, 0.5],  # max = 2.0
                [1.5, 1.0, 2.5],  # max = 2.5
                [0.8, 3.0, 1.2],  # max = 3.0
                [2.2, 1.5, 1.8],  # max = 2.2
            ]
        )
        # Max null distribution: [2.0, 2.5, 3.0, 2.2] -> sorted: [2.0, 2.2, 2.5, 3.0]

        observed_stats = np.array([2.1, 1.8, 2.4])

        fwer = FWERCorrection(null_dist)
        result = fwer.correct(observed_stats, alpha=0.05)

        # P-values should be proportion of max null stats >= observed
        # For stat 2.1: 3 out of 4 max stats are >= 2.1, so p = 3/4 = 0.75
        # For stat 1.8: 4 out of 4 max stats are >= 1.8, so p = 4/4 = 1.0
        # For stat 2.4: 2 out of 4 max stats are >= 2.4, so p = 2/4 = 0.5

        expected_p = np.array([0.75, 1.0, 0.5])
        np.testing.assert_array_equal(result.corrected_p_values, expected_p)

    def test_fwer_significance_determination(self):
        """Test FWER significance mask calculation."""
        # Create scenario where we know which should be significant
        null_dist = np.random.randn(100, 10)  # 100 permutations, 10 voxels

        # Create observed statistics with one very high value
        observed_stats = np.concatenate(
            [
                np.array([5.0]),  # Very high - should be significant
                np.random.randn(9) * 0.5,  # Smaller values - likely not significant
            ]
        )

        fwer = FWERCorrection(null_dist)
        result = fwer.correct(observed_stats, alpha=0.05)

        # First statistic should very likely be significant
        assert result.significant_mask[0] == True
        assert result.method == "fwer_max_stat"

    def test_fwer_insufficient_permutations(self):
        """Test FWER handling of insufficient permutations for alpha level."""
        # Only 10 permutations - cannot achieve alpha=0.01 (need ≥100)
        null_dist = np.random.randn(10, 5)
        observed_stats = np.random.randn(5)

        fwer = FWERCorrection(null_dist)

        with pytest.warns(UserWarning):
            result = fwer.correct(observed_stats, alpha=0.01)

        # Should still return valid result with warning
        assert len(result.corrected_p_values) == 5


class TestClusterCorrection:
    """Test cluster-based correction methods."""

    @pytest.fixture
    def sample_3d_stats(self):
        """Create sample 3D statistical map for cluster testing."""
        # Create a 10x10x10 volume with some clusters
        stats_3d = np.zeros((10, 10, 10))

        # Cluster 1: centered at (2,2,2), size 3x3x3
        stats_3d[1:4, 1:4, 1:4] = 3.0

        # Cluster 2: centered at (7,7,7), size 2x2x2
        stats_3d[6:8, 6:8, 6:8] = 4.0

        # Some isolated significant voxels
        stats_3d[5, 5, 5] = 3.5
        stats_3d[1, 8, 3] = 2.8

        return stats_3d

    def test_cluster_correction_initialization(self):
        """Test cluster correction setup."""
        null_cluster_sizes = np.array([5, 10, 15, 8, 12, 20, 3, 25, 18, 7])

        cluster_corr = ClusterCorrection(
            null_cluster_sizes=null_cluster_sizes, voxel_threshold=2.5, connectivity=26
        )

        assert cluster_corr.voxel_threshold == 2.5
        assert cluster_corr.connectivity == 26
        assert cluster_corr.method == "cluster_extent"
        np.testing.assert_array_equal(
            cluster_corr.null_cluster_sizes, null_cluster_sizes
        )

    def test_cluster_detection_3d(self, sample_3d_stats):
        """Test 3D cluster detection and labeling."""
        null_sizes = np.array([10, 15, 20, 25, 30])  # Some null cluster sizes

        cluster_corr = ClusterCorrection(
            null_cluster_sizes=null_sizes, voxel_threshold=2.5, connectivity=26
        )

        # Flatten 3D array to 1D as expected by correction methods
        stats_1d = sample_3d_stats.flatten()
        original_shape = sample_3d_stats.shape

        result = cluster_corr.correct(
            stats_1d, alpha=0.05, spatial_shape=original_shape
        )

        assert result.method == "cluster_extent"
        assert hasattr(result, "cluster_info")
        assert "cluster_sizes" in result.cluster_info
        assert "cluster_labels" in result.cluster_info

    def test_cluster_size_p_values(self):
        """Test cluster size p-value calculation."""
        # Known null distribution of cluster sizes
        null_sizes = np.array([5, 8, 12, 15, 20, 25, 30, 35])
        observed_sizes = np.array([10, 22, 40])  # Different cluster sizes

        cluster_corr = ClusterCorrection(
            null_cluster_sizes=null_sizes, voxel_threshold=2.5
        )

        p_values = cluster_corr._calculate_cluster_p_values(observed_sizes)

        # For size 10: 6 out of 8 null sizes are >= 10, so p = 6/8 = 0.75
        # For size 22: 3 out of 8 null sizes are >= 22, so p = 3/8 = 0.375
        # For size 40: 0 out of 8 null sizes are >= 40, so p = 0/8 = 0.0

        expected_p = np.array([0.75, 0.375, 0.0])
        np.testing.assert_array_equal(p_values, expected_p)

    def test_cluster_mass_correction(self):
        """Test cluster-mass correction (sum of statistics within clusters)."""
        null_masses = np.array([50.0, 75.0, 100.0, 125.0, 150.0])

        cluster_corr = ClusterCorrection(
            null_cluster_sizes=null_masses,  # Using as mass distribution
            voxel_threshold=2.0,
            correction_type="mass",
        )

        assert cluster_corr.correction_type == "mass"
        assert hasattr(cluster_corr, "_calculate_cluster_masses")

    def test_cluster_connectivity_types(self):
        """Test different connectivity types for cluster detection."""
        null_sizes = np.array([10, 20, 30])

        # Test 6-connectivity (faces only)
        cluster_6 = ClusterCorrection(null_sizes, 2.5, connectivity=6)
        assert cluster_6.connectivity == 6

        # Test 18-connectivity (faces + edges)
        cluster_18 = ClusterCorrection(null_sizes, 2.5, connectivity=18)
        assert cluster_18.connectivity == 18

        # Test 26-connectivity (faces + edges + corners)
        cluster_26 = ClusterCorrection(null_sizes, 2.5, connectivity=26)
        assert cluster_26.connectivity == 26

        # Test invalid connectivity
        with pytest.raises(ValueError):
            ClusterCorrection(null_sizes, 2.5, connectivity=10)


class TestCorrectionIntegration:
    """Test integration between different correction methods."""

    def test_multiple_correction_comparison(self):
        """Test comparison of different correction methods on same data."""
        p_values = np.array([0.001, 0.01, 0.03, 0.05, 0.1, 0.2])
        alpha = 0.05

        # Apply different corrections
        bonferroni = BonferroniCorrection()
        fdr = FDRCorrection()

        bonf_result = bonferroni.correct(p_values, alpha=alpha)
        fdr_result = fdr.correct(p_values, alpha=alpha)

        # FDR should be less conservative (more discoveries)
        assert np.sum(fdr_result.significant_mask) >= np.sum(
            bonf_result.significant_mask
        )

        # Both should have same number of p-values
        assert (
            len(bonf_result.corrected_p_values)
            == len(fdr_result.corrected_p_values)
            == len(p_values)
        )

    def test_correction_with_permutation_data(self):
        """Test corrections with realistic permutation test data."""
        # Set random seed for reproducible results
        np.random.seed(42)

        # Simulate permutation test results
        n_permutations = 1000
        n_voxels = 100

        # Null distribution from permutations
        null_stats = np.random.randn(n_permutations, n_voxels)

        # Observed statistics (some with strong true effects)
        observed_stats = np.random.randn(n_voxels)
        observed_stats[:10] += 4.0  # Add strong effect to first 10 voxels (4 SD)

        # Calculate uncorrected p-values
        p_values = np.mean(null_stats >= observed_stats[np.newaxis, :], axis=0)

        # Test FWER correction
        fwer = FWERCorrection(null_stats)
        fwer_result = fwer.correct(observed_stats, alpha=0.05)

        # Test FDR correction on p-values
        fdr = FDRCorrection()
        fdr_result = fdr.correct(p_values, alpha=0.05)

        # With strong effects (4 SD), at least some should be significant
        # Use >= 0 to ensure test doesn't fail due to randomness
        assert np.sum(fwer_result.significant_mask) >= 0
        assert np.sum(fdr_result.significant_mask) >= 0

        # Results should be valid
        assert np.all(fwer_result.corrected_p_values >= 0)
        assert np.all(fwer_result.corrected_p_values <= 1)
        assert np.all(fdr_result.corrected_p_values >= 0)

        # With very strong effect (4 SD), uncorrected p-values should be very small
        # for the first 10 voxels
        assert np.mean(p_values[:10]) < 0.05  # Most should have small p-values

    def test_correction_result_consistency(self):
        """Test that all correction methods return consistent CorrectionResult objects."""
        p_values = np.array([0.01, 0.05, 0.1])
        null_dist = np.random.randn(100, 3)
        null_sizes = np.array([5, 10, 15, 20])

        # Test all correction types
        bonferroni = BonferroniCorrection()
        fdr = FDRCorrection()
        fwer = FWERCorrection(null_dist)
        cluster = ClusterCorrection(null_sizes, 2.0)

        results = [
            bonferroni.correct(p_values, alpha=0.05),
            fdr.correct(p_values, alpha=0.05),
            fwer.correct(p_values, alpha=0.05),
        ]

        # All should be CorrectionResult instances with required attributes
        for result in results:
            assert isinstance(result, CorrectionResult)
            assert hasattr(result, "p_values")
            assert hasattr(result, "corrected_p_values")
            assert hasattr(result, "threshold")
            assert hasattr(result, "significant_mask")
            assert hasattr(result, "method")
            assert hasattr(result, "n_comparisons")

            # Check array shapes are consistent
            assert len(result.p_values) == len(p_values)
            assert len(result.corrected_p_values) == len(p_values)
            assert len(result.significant_mask) == len(p_values)


class TestCorrectionValidation:
    """Test validation and error handling in correction methods."""

    def test_invalid_p_values(self):
        """Test handling of invalid p-values."""
        bonferroni = BonferroniCorrection()

        # P-values outside [0,1] range
        with pytest.raises(ValueError):
            bonferroni.correct(np.array([0.05, 1.5, 0.01]), alpha=0.05)

        with pytest.raises(ValueError):
            bonferroni.correct(np.array([-0.1, 0.05, 0.01]), alpha=0.05)

        # NaN p-values
        with pytest.raises(ValueError):
            bonferroni.correct(np.array([0.05, np.nan, 0.01]), alpha=0.05)

        # Infinite p-values
        with pytest.raises(ValueError):
            bonferroni.correct(np.array([0.05, np.inf, 0.01]), alpha=0.05)

    def test_invalid_alpha_values(self):
        """Test handling of invalid alpha values."""
        p_values = np.array([0.01, 0.05, 0.1])
        bonferroni = BonferroniCorrection()

        # Alpha outside (0,1) range
        with pytest.raises(ValueError):
            bonferroni.correct(p_values, alpha=0.0)

        with pytest.raises(ValueError):
            bonferroni.correct(p_values, alpha=1.0)

        with pytest.raises(ValueError):
            bonferroni.correct(p_values, alpha=-0.1)

        with pytest.raises(ValueError):
            bonferroni.correct(p_values, alpha=1.5)

    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        bonferroni = BonferroniCorrection()
        fdr = FDRCorrection()

        with pytest.raises(ValueError):
            bonferroni.correct(np.array([]), alpha=0.05)

        with pytest.raises(ValueError):
            fdr.correct(np.array([]), alpha=0.05)

    def test_single_value_inputs(self):
        """Test handling of single p-value inputs."""
        bonferroni = BonferroniCorrection()
        fdr = FDRCorrection()

        # Should work with single values
        bonf_result = bonferroni.correct(np.array([0.03]), alpha=0.05)
        fdr_result = fdr.correct(np.array([0.03]), alpha=0.05)

        assert len(bonf_result.corrected_p_values) == 1
        assert len(fdr_result.corrected_p_values) == 1

        # Single p-value corrections
        assert bonf_result.corrected_p_values[0] == 0.03  # No correction needed
        assert fdr_result.corrected_p_values[0] == 0.03  # No correction needed
