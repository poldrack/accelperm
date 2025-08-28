"""Tests for advanced permutation strategies - TDD RED phase."""

import numpy as np
import pytest

from accelperm.core.advanced_permutation import (
    AdaptiveMonteCarlo,
    FullEnumerationDetector,
    TwoStagePermutation,
    VarianceSmoothing,
)


class TestAdaptiveMonteCarlo:
    """Test adaptive Monte Carlo sampling strategy - RED phase."""

    def test_adaptive_monte_carlo_exists(self):
        """Test that AdaptiveMonteCarlo class exists - RED phase."""
        # This should fail because AdaptiveMonteCarlo doesn't exist yet
        mc = AdaptiveMonteCarlo(initial_samples=1000, confidence_level=0.95)
        assert mc.initial_samples == 1000
        assert mc.confidence_level == 0.95

    def test_monte_carlo_sample_size_adaptation(self):
        """Test sample size adaptation based on statistical properties - RED phase."""
        mc = AdaptiveMonteCarlo(initial_samples=100, confidence_level=0.95)

        # Simulate some test statistics
        test_stats = np.random.randn(100) * 2.5

        # Should suggest increased sample size for high variance
        recommended_size = mc.recommend_sample_size(test_stats)
        assert recommended_size >= 100
        assert isinstance(recommended_size, int)

    def test_monte_carlo_convergence_detection(self):
        """Test convergence detection for p-values - RED phase."""
        mc = AdaptiveMonteCarlo(initial_samples=500)

        # Create synthetic p-value distribution
        p_values = np.random.beta(2, 8, 500)  # Skewed toward small p-values

        # Should detect when p-value estimate has converged
        is_converged, margin_of_error = mc.check_convergence(p_values)
        assert isinstance(is_converged, bool)
        assert isinstance(margin_of_error, float)
        assert margin_of_error >= 0

    def test_monte_carlo_batch_processing(self):
        """Test batch processing of permutations - RED phase."""
        mc = AdaptiveMonteCarlo(initial_samples=1000, batch_size=100)

        # Should generate batches that sum to target
        batches = list(mc.generate_batches())
        total_samples = sum(len(batch) for batch in batches)
        assert total_samples == 1000

        # Each batch should be reasonably sized
        for batch in batches:
            assert len(batch) <= 100

    def test_monte_carlo_early_stopping(self):
        """Test early stopping when sufficient precision achieved - RED phase."""
        mc = AdaptiveMonteCarlo(initial_samples=10000, target_precision=0.01)

        # Simulate gradually improving precision
        p_estimates = [0.05, 0.048, 0.051, 0.0495, 0.0505]  # Converging to 0.05

        should_stop = mc.should_stop_early(p_estimates, current_samples=5000)
        assert isinstance(should_stop, bool)


class TestFullEnumerationDetector:
    """Test full enumeration detection and switching - RED phase."""

    def test_full_enumeration_detector_exists(self):
        """Test that FullEnumerationDetector class exists - RED phase."""
        # This should fail because FullEnumerationDetector doesn't exist yet
        detector = FullEnumerationDetector(max_permutations=10000)
        assert detector.max_permutations == 10000

    def test_detect_feasible_enumeration(self):
        """Test detection of feasible full enumeration - RED phase."""
        detector = FullEnumerationDetector(max_permutations=5000)

        # Small sample - should recommend enumeration
        n_subjects = 8
        is_feasible, n_possible = detector.is_enumeration_feasible(n_subjects)
        assert isinstance(is_feasible, bool)
        assert isinstance(n_possible, int)

        # For 8 subjects: 8! = 40320, might be too large
        # For 6 subjects: 6! = 720, should be feasible
        n_subjects = 6
        is_feasible_small, n_possible_small = detector.is_enumeration_feasible(
            n_subjects
        )
        assert is_feasible_small is True
        assert n_possible_small == 720

    def test_memory_estimation(self):
        """Test memory requirement estimation - RED phase."""
        detector = FullEnumerationDetector()

        # Should estimate memory needed for enumeration
        memory_gb = detector.estimate_memory_requirement(n_subjects=7, dtype_size=8)
        assert isinstance(memory_gb, float)
        assert memory_gb > 0

        # Larger samples should require more memory
        memory_gb_large = detector.estimate_memory_requirement(
            n_subjects=10, dtype_size=8
        )
        assert memory_gb_large > memory_gb

    def test_automatic_strategy_selection(self):
        """Test automatic selection between enumeration and sampling - RED phase."""
        detector = FullEnumerationDetector()

        # Should select enumeration for small samples
        strategy = detector.recommend_strategy(n_subjects=6, n_requested=1000)
        assert strategy in ["enumeration", "sampling"]

        # Should select sampling for large samples
        strategy_large = detector.recommend_strategy(n_subjects=15, n_requested=10000)
        assert strategy_large == "sampling"


class TestVarianceSmoothing:
    """Test variance smoothing for improved statistical power - RED phase."""

    def test_variance_smoothing_exists(self):
        """Test that VarianceSmoothing class exists - RED phase."""
        # This should fail because VarianceSmoothing doesn't exist yet
        smoother = VarianceSmoothing(method="empirical_bayes")
        assert smoother.method == "empirical_bayes"

    def test_smooth_variance_estimates(self):
        """Test smoothing of variance estimates - RED phase."""
        smoother = VarianceSmoothing(method="empirical_bayes")

        # Create synthetic variance estimates (some very small)
        raw_variances = np.array([0.1, 0.001, 0.05, 0.2, 0.0001, 0.15])

        # Should return smoothed variances
        smoothed_variances = smoother.smooth_variances(raw_variances)
        assert smoothed_variances.shape == raw_variances.shape
        assert np.all(smoothed_variances > 0)

        # Smoothed variances should be less extreme
        assert np.min(smoothed_variances) > np.min(raw_variances)

    def test_empirical_bayes_smoothing(self):
        """Test empirical Bayes variance smoothing - RED phase."""
        smoother = VarianceSmoothing(method="empirical_bayes")

        # Create realistic variance pattern
        n_voxels = 1000
        true_variances = np.random.gamma(2, 0.1, n_voxels)  # Gamma-distributed
        raw_variances = true_variances + np.random.normal(0, 0.01, n_voxels)

        # Should shrink toward prior
        smoothed = smoother.smooth_variances(raw_variances)

        # Check that extreme values are moderated
        extreme_low = raw_variances < np.percentile(raw_variances, 5)
        extreme_high = raw_variances > np.percentile(raw_variances, 95)

        # Smoothed extreme values should be closer to the mean
        if np.any(extreme_low):
            assert np.mean(smoothed[extreme_low]) > np.mean(raw_variances[extreme_low])
        if np.any(extreme_high):
            assert np.mean(smoothed[extreme_high]) < np.mean(
                raw_variances[extreme_high]
            )

    def test_smoothing_with_covariates(self):
        """Test variance smoothing with covariate information - RED phase."""
        smoother = VarianceSmoothing(method="covariate_adjustment")

        # Create synthetic data with covariate structure
        n_voxels = 500
        brain_region = np.random.randint(0, 5, n_voxels)  # 5 brain regions
        raw_variances = np.random.exponential(0.1, n_voxels)

        # Should incorporate covariate information
        smoothed = smoother.smooth_variances(
            raw_variances, covariates={"brain_region": brain_region}
        )
        assert smoothed.shape == raw_variances.shape
        assert np.all(smoothed > 0)

    def test_degrees_of_freedom_adjustment(self):
        """Test degrees of freedom adjustment with smoothing - RED phase."""
        smoother = VarianceSmoothing()

        raw_variances = np.random.exponential(0.2, 100)
        original_df = 20

        # Should return adjusted degrees of freedom
        smoothed_variances, adjusted_df = smoother.smooth_with_df_adjustment(
            raw_variances, original_df
        )

        assert smoothed_variances.shape == raw_variances.shape
        assert isinstance(adjusted_df, (int, float))
        assert adjusted_df >= original_df  # Should increase effective df


class TestTwoStagePermutation:
    """Test two-stage permutation for complex designs - RED phase."""

    def test_two_stage_permutation_exists(self):
        """Test that TwoStagePermutation class exists - RED phase."""
        # This should fail because TwoStagePermutation doesn't exist yet
        two_stage = TwoStagePermutation(
            stage1_factor="subject", stage2_factor="condition"
        )
        assert two_stage.stage1_factor == "subject"
        assert two_stage.stage2_factor == "condition"

    def test_nested_permutation_structure(self):
        """Test nested permutation structure handling - RED phase."""
        two_stage = TwoStagePermutation(
            stage1_factor="subject", stage2_factor="session"
        )

        # Create hierarchical design
        n_subjects = 10
        n_sessions_per_subject = 3
        n_observations = n_subjects * n_sessions_per_subject

        subject_ids = np.repeat(range(n_subjects), n_sessions_per_subject)
        session_ids = np.tile(range(n_sessions_per_subject), n_subjects)

        # Should handle nested structure
        perm_scheme = two_stage.create_permutation_scheme(
            subject_ids=subject_ids, session_ids=session_ids
        )

        assert "stage1_blocks" in perm_scheme
        assert "stage2_blocks" in perm_scheme
        assert len(perm_scheme["stage1_blocks"]) == n_subjects

    def test_between_within_subject_permutation(self):
        """Test between-subject and within-subject permutation - RED phase."""
        two_stage = TwoStagePermutation()

        # Mixed-effects design: between-subject factor + within-subject measurements
        n_subjects = 20
        n_timepoints = 4

        # Between-subject factor: group assignment
        group = np.random.binomial(1, 0.5, n_subjects)

        # Within-subject factor: timepoints
        design = {
            "subject_ids": np.repeat(range(n_subjects), n_timepoints),
            "group": np.repeat(group, n_timepoints),
            "timepoint": np.tile(range(n_timepoints), n_subjects),
        }

        # Should generate proper two-stage permutation
        permutation = two_stage.generate_permutation(design)

        assert "between_subject_perm" in permutation
        assert "within_subject_perm" in permutation
        assert len(permutation["between_subject_perm"]) == n_subjects

    def test_restricted_randomization(self):
        """Test restricted randomization within blocks - RED phase."""
        two_stage = TwoStagePermutation()

        # Create blocks for restricted randomization
        n_subjects = 16
        block_size = 4
        blocks = np.repeat(range(n_subjects // block_size), block_size)

        # Should maintain balance within blocks
        permutation = two_stage.generate_blocked_permutation(
            n_subjects=n_subjects, blocks=blocks, n_permutations=100
        )

        assert permutation.shape == (n_subjects, 100)

        # Check that permutations respect block structure
        for perm_idx in range(100):
            perm = permutation[:, perm_idx]
            for block_id in range(len(np.unique(blocks))):
                block_indices = np.where(blocks == block_id)[0]
                block_perm = perm[block_indices]
                # Block should be a valid permutation of its indices
                assert set(block_perm) == set(block_indices)


class TestAdvancedPermutationIntegration:
    """Test integration between advanced permutation strategies - RED phase."""

    def test_adaptive_enumeration_switching(self):
        """Test automatic switching between enumeration and sampling - RED phase."""
        # Should use enumeration for small samples
        small_detector = FullEnumerationDetector(max_permutations=1000)
        small_mc = AdaptiveMonteCarlo(initial_samples=500)

        n_subjects = 6  # 6! = 720 permutations
        strategy = small_detector.recommend_strategy(n_subjects, n_requested=1000)

        if strategy == "enumeration":
            # Should use all possible permutations
            assert small_detector.is_enumeration_feasible(n_subjects)[1] <= 1000
        else:
            # Should use Monte Carlo
            assert isinstance(small_mc.initial_samples, int)

    def test_smoothed_statistics_with_permutation(self):
        """Test variance smoothing integration with permutation testing - RED phase."""
        smoother = VarianceSmoothing()

        # Create synthetic data
        n_voxels, n_subjects = 100, 20
        Y = np.random.randn(n_voxels, n_subjects)
        X = np.random.randn(n_subjects, 3)

        # Compute raw statistics
        raw_variances = np.var(Y, axis=1)

        # Should integrate with permutation engine
        smoothed_variances = smoother.smooth_variances(raw_variances)

        # Smoothed variances should be used in permutation testing
        assert len(smoothed_variances) == n_voxels
        assert np.all(smoothed_variances > 0)

    def test_two_stage_with_monte_carlo(self):
        """Test two-stage permutation with Monte Carlo sampling - RED phase."""
        two_stage = TwoStagePermutation()
        mc = AdaptiveMonteCarlo(initial_samples=500)

        # Complex hierarchical design
        n_subjects = 15
        n_sessions = 2

        design = {
            "subject_ids": np.repeat(range(n_subjects), n_sessions),
            "session_ids": np.tile(range(n_sessions), n_subjects),
        }

        # Should efficiently handle complex permutation structure
        perm_scheme = two_stage.create_permutation_scheme(
            subject_ids=design["subject_ids"], session_ids=design["session_ids"]
        )

        # Should recommend appropriate sampling strategy
        recommended_samples = mc.recommend_sample_size([])  # Empty to get default
        assert isinstance(recommended_samples, int)
        assert recommended_samples >= mc.initial_samples
