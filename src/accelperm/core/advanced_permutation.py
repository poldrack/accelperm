"""Advanced permutation strategies for neuroimaging statistical analysis.

This module provides sophisticated permutation testing strategies including:
- Adaptive Monte Carlo sampling with convergence detection
- Full enumeration detection and automatic switching
- Variance smoothing for improved statistical power
- Two-stage permutation for complex hierarchical designs

Based on FSL randomise advanced features with GPU acceleration support.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
from scipy import stats

from accelperm.utils.logging import LogLevel, setup_logger

logger = setup_logger(__name__, LogLevel.INFO)


class AdaptiveMonteCarlo:
    """Adaptive Monte Carlo sampling with convergence detection."""

    def __init__(
        self,
        initial_samples: int = 1000,
        confidence_level: float = 0.95,
        batch_size: int = 100,
        target_precision: float = 0.01,
        max_samples: int = 100000,
    ):
        """Initialize adaptive Monte Carlo sampler.

        Parameters
        ----------
        initial_samples : int
            Initial number of samples to draw.
        confidence_level : float
            Confidence level for convergence testing.
        batch_size : int
            Size of each sampling batch.
        target_precision : float
            Target precision for p-value estimates.
        max_samples : int
            Maximum number of samples to draw.
        """
        self.initial_samples = initial_samples
        self.confidence_level = confidence_level
        self.batch_size = batch_size
        self.target_precision = target_precision
        self.max_samples = max_samples

        self.z_critical = stats.norm.ppf((1 + confidence_level) / 2)

    def recommend_sample_size(self, test_statistics: np.ndarray | None = None) -> int:
        """Recommend sample size based on test statistics properties.

        Parameters
        ----------
        test_statistics : np.ndarray | None
            Array of test statistics to analyze.

        Returns
        -------
        int
            Recommended number of samples.
        """
        if test_statistics is None or len(test_statistics) == 0:
            return self.initial_samples

        # Estimate variance in test statistics
        stat_variance = np.var(test_statistics)

        # Higher variance suggests need for more samples
        # Use coefficient of variation to scale recommendation
        cv = np.sqrt(stat_variance) / (np.abs(np.mean(test_statistics)) + 1e-10)

        # Scale initial samples based on coefficient of variation
        scale_factor = min(5.0, max(1.0, cv))
        recommended = int(self.initial_samples * scale_factor)

        return min(recommended, self.max_samples)

    def check_convergence(self, p_values: np.ndarray) -> tuple[bool, float]:
        """Check if p-value estimates have converged.

        Parameters
        ----------
        p_values : np.ndarray
            Array of p-values from permutation samples.

        Returns
        -------
        tuple[bool, float]
            Whether convergence achieved and margin of error.
        """
        if len(p_values) < 10:
            return False, float("inf")

        # Estimate p-value
        p_estimate = np.mean(p_values)

        # Handle edge cases
        if p_estimate == 0:
            p_estimate = 1 / len(p_values)  # Conservative estimate
        elif p_estimate == 1:
            p_estimate = 1 - 1 / len(p_values)

        # Standard error for proportion
        se = np.sqrt(p_estimate * (1 - p_estimate) / len(p_values))

        # Margin of error
        margin_of_error = self.z_critical * se

        # Check if precision target is met
        is_converged = bool(margin_of_error <= self.target_precision)

        return is_converged, margin_of_error

    def generate_batches(self) -> Iterator[range]:
        """Generate sample batches for processing.

        Yields
        ------
        range
            Range object for each batch.
        """
        total_generated = 0

        while total_generated < self.initial_samples:
            batch_end = min(total_generated + self.batch_size, self.initial_samples)
            yield range(total_generated, batch_end)
            total_generated = batch_end

    def should_stop_early(self, p_estimates: list[float], current_samples: int) -> bool:
        """Determine if sampling should stop early.

        Parameters
        ----------
        p_estimates : list[float]
            List of p-value estimates from recent batches.
        current_samples : int
            Current number of samples processed.

        Returns
        -------
        bool
            Whether to stop sampling early.
        """
        if len(p_estimates) < 3 or current_samples < self.initial_samples // 2:
            return False

        # Check if recent estimates are stable
        recent_estimates = p_estimates[-3:]
        stability = np.std(recent_estimates)

        # Stop if estimates are stable and we have sufficient samples
        is_stable = stability < self.target_precision / 2
        has_min_samples = current_samples >= self.initial_samples // 2

        return is_stable and has_min_samples


class FullEnumerationDetector:
    """Detector for feasible full enumeration scenarios."""

    def __init__(
        self,
        max_permutations: int = 10000,
        max_memory_gb: float = 8.0,
        memory_safety_factor: float = 0.8,
    ):
        """Initialize enumeration detector.

        Parameters
        ----------
        max_permutations : int
            Maximum permutations to consider feasible.
        max_memory_gb : float
            Maximum memory to use in GB.
        memory_safety_factor : float
            Safety factor for memory estimation.
        """
        self.max_permutations = max_permutations
        self.max_memory_gb = max_memory_gb
        self.memory_safety_factor = memory_safety_factor

    def is_enumeration_feasible(self, n_subjects: int) -> tuple[bool, int]:
        """Check if full enumeration is feasible.

        Parameters
        ----------
        n_subjects : int
            Number of subjects in the study.

        Returns
        -------
        tuple[bool, int]
            Whether enumeration is feasible and number of possible permutations.
        """
        if n_subjects > 20:  # Factorial grows too quickly
            return False, math.factorial(n_subjects)

        n_possible = math.factorial(n_subjects)

        # Check computational feasibility
        computationally_feasible = n_possible <= self.max_permutations

        # Check memory feasibility
        memory_required = self.estimate_memory_requirement(n_subjects)
        memory_feasible = memory_required <= self.max_memory_gb

        is_feasible = computationally_feasible and memory_feasible

        return is_feasible, n_possible

    def estimate_memory_requirement(
        self, n_subjects: int, dtype_size: int = 8
    ) -> float:
        """Estimate memory requirement for full enumeration.

        Parameters
        ----------
        n_subjects : int
            Number of subjects.
        dtype_size : int
            Size of data type in bytes.

        Returns
        -------
        float
            Estimated memory requirement in GB.
        """
        if n_subjects > 20:
            # Return a large number to indicate infeasibility
            return float("inf")

        n_permutations = math.factorial(n_subjects)

        # Memory for permutation matrix
        perm_matrix_bytes = n_subjects * n_permutations * dtype_size

        # Memory for statistics storage (assume similar size)
        stats_storage_bytes = perm_matrix_bytes

        # Total memory with safety factor
        total_bytes = (
            perm_matrix_bytes + stats_storage_bytes
        ) / self.memory_safety_factor

        # Convert to GB
        return total_bytes / (1024**3)

    def recommend_strategy(
        self, n_subjects: int, n_requested: int
    ) -> Literal["enumeration", "sampling"]:
        """Recommend strategy based on problem size.

        Parameters
        ----------
        n_subjects : int
            Number of subjects.
        n_requested : int
            Requested number of permutations.

        Returns
        -------
        Literal["enumeration", "sampling"]
            Recommended strategy.
        """
        is_feasible, n_possible = self.is_enumeration_feasible(n_subjects)

        if not is_feasible:
            return "sampling"

        # If we can enumerate all and it's less than requested, use enumeration
        if n_possible <= n_requested:
            return "enumeration"

        # If enumeration is feasible but we need fewer permutations, still use sampling
        # unless the difference is small
        if n_requested < n_possible * 0.5:
            return "sampling"
        else:
            return "enumeration"


class VarianceSmoothing:
    """Variance smoothing for improved statistical power."""

    def __init__(
        self,
        method: Literal[
            "empirical_bayes", "covariate_adjustment", "simple"
        ] = "empirical_bayes",
        prior_df: float = 4.0,
        shrinkage_intensity: float = 0.1,
    ):
        """Initialize variance smoother.

        Parameters
        ----------
        method : Literal["empirical_bayes", "covariate_adjustment", "simple"]
            Smoothing method to use.
        prior_df : float
            Prior degrees of freedom for empirical Bayes.
        shrinkage_intensity : float
            Intensity of shrinkage toward prior.
        """
        self.method = method
        self.prior_df = prior_df
        self.shrinkage_intensity = shrinkage_intensity

    def smooth_variances(
        self, raw_variances: np.ndarray, covariates: dict[str, np.ndarray] | None = None
    ) -> np.ndarray:
        """Smooth variance estimates.

        Parameters
        ----------
        raw_variances : np.ndarray
            Raw variance estimates.
        covariates : dict[str, np.ndarray] | None
            Covariate information for smoothing.

        Returns
        -------
        np.ndarray
            Smoothed variance estimates.
        """
        if self.method == "empirical_bayes":
            return self._empirical_bayes_smoothing(raw_variances)
        elif self.method == "covariate_adjustment":
            return self._covariate_adjustment_smoothing(raw_variances, covariates)
        elif self.method == "simple":
            return self._simple_smoothing(raw_variances)
        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")

    def _empirical_bayes_smoothing(self, raw_variances: np.ndarray) -> np.ndarray:
        """Apply empirical Bayes smoothing."""
        # Estimate prior parameters
        log_vars = np.log(np.maximum(raw_variances, 1e-10))
        prior_mean = np.mean(log_vars)

        # Shrink toward prior
        shrinkage_factor = self.shrinkage_intensity
        smoothed_log_vars = (
            1 - shrinkage_factor
        ) * log_vars + shrinkage_factor * prior_mean

        # Transform back
        smoothed_variances = np.exp(smoothed_log_vars)

        # Ensure minimum variance
        min_variance = np.percentile(raw_variances, 1)
        smoothed_variances = np.maximum(smoothed_variances, min_variance)

        return smoothed_variances

    def _covariate_adjustment_smoothing(
        self, raw_variances: np.ndarray, covariates: dict[str, np.ndarray] | None
    ) -> np.ndarray:
        """Apply covariate-based smoothing."""
        if covariates is None:
            return self._simple_smoothing(raw_variances)

        smoothed = raw_variances.copy()

        # For each covariate, smooth within groups
        for _covar_name, covar_values in covariates.items():
            unique_values = np.unique(covar_values)

            for value in unique_values:
                mask = covar_values == value
                if np.sum(mask) > 1:
                    group_vars = raw_variances[mask]
                    group_mean = np.mean(group_vars)

                    # Shrink within group
                    smoothed[mask] = (
                        1 - self.shrinkage_intensity
                    ) * group_vars + self.shrinkage_intensity * group_mean

        return smoothed

    def _simple_smoothing(self, raw_variances: np.ndarray) -> np.ndarray:
        """Apply simple smoothing toward global mean."""
        global_mean = np.mean(raw_variances)

        smoothed = (
            1 - self.shrinkage_intensity
        ) * raw_variances + self.shrinkage_intensity * global_mean

        return smoothed

    def smooth_with_df_adjustment(
        self, raw_variances: np.ndarray, original_df: int
    ) -> tuple[np.ndarray, float]:
        """Smooth variances and adjust degrees of freedom.

        Parameters
        ----------
        raw_variances : np.ndarray
            Raw variance estimates.
        original_df : int
            Original degrees of freedom.

        Returns
        -------
        tuple[np.ndarray, float]
            Smoothed variances and adjusted degrees of freedom.
        """
        smoothed_variances = self.smooth_variances(raw_variances)

        # Adjust degrees of freedom based on smoothing
        # More smoothing effectively increases degrees of freedom
        df_adjustment = self.prior_df * self.shrinkage_intensity
        adjusted_df = original_df + df_adjustment

        return smoothed_variances, adjusted_df


class TwoStagePermutation:
    """Two-stage permutation for complex hierarchical designs."""

    def __init__(
        self,
        stage1_factor: str = "subject",
        stage2_factor: str = "condition",
        seed: int | None = None,
    ):
        """Initialize two-stage permutation handler.

        Parameters
        ----------
        stage1_factor : str
            First-stage factor (typically subject).
        stage2_factor : str
            Second-stage factor (typically condition/session).
        seed : int | None
            Random seed for reproducibility.
        """
        self.stage1_factor = stage1_factor
        self.stage2_factor = stage2_factor
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def create_permutation_scheme(
        self,
        subject_ids: np.ndarray,
        session_ids: np.ndarray | None = None,
        **kwargs: np.ndarray,
    ) -> dict[str, Any]:
        """Create permutation scheme for hierarchical design.

        Parameters
        ----------
        subject_ids : np.ndarray
            Subject identifier for each observation.
        session_ids : np.ndarray | None
            Session identifier for each observation.
        **kwargs : np.ndarray
            Additional factor arrays.

        Returns
        -------
        dict[str, Any]
            Permutation scheme configuration.
        """
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)

        # Create stage 1 blocks (subjects)
        stage1_blocks = {}
        for subj in unique_subjects:
            stage1_blocks[subj] = np.where(subject_ids == subj)[0]

        # Create stage 2 blocks (within-subject factors)
        stage2_blocks = {}
        if session_ids is not None:
            for subj in unique_subjects:
                subj_mask = subject_ids == subj
                subj_sessions = session_ids[subj_mask]
                unique_sessions = np.unique(subj_sessions)

                stage2_blocks[subj] = {}
                for sess in unique_sessions:
                    sess_mask = (subject_ids == subj) & (session_ids == sess)
                    stage2_blocks[subj][sess] = np.where(sess_mask)[0]

        return {
            "stage1_blocks": stage1_blocks,
            "stage2_blocks": stage2_blocks,
            "n_subjects": n_subjects,
            "factors": {
                "subject_ids": subject_ids,
                "session_ids": session_ids,
                **kwargs,
            },
        }

    def generate_permutation(
        self, design: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Generate two-stage permutation.

        Parameters
        ----------
        design : dict[str, np.ndarray]
            Design specification with factor arrays.

        Returns
        -------
        dict[str, np.ndarray]
            Two-stage permutation specification.
        """
        subject_ids = design["subject_ids"]
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)

        # Stage 1: Between-subject permutation
        between_subject_perm = self.rng.permutation(unique_subjects)

        # Stage 2: Within-subject permutation (if applicable)
        within_subject_perm = {}
        if "timepoint" in design:
            timepoints = design["timepoint"]
            unique_timepoints = np.unique(timepoints)

            for subj in unique_subjects:
                # Permute timepoints within subject
                within_subject_perm[subj] = self.rng.permutation(unique_timepoints)

        return {
            "between_subject_perm": between_subject_perm,
            "within_subject_perm": within_subject_perm,
            "n_subjects": n_subjects,
        }

    def generate_blocked_permutation(
        self, n_subjects: int, blocks: np.ndarray, n_permutations: int
    ) -> np.ndarray:
        """Generate blocked permutations.

        Parameters
        ----------
        n_subjects : int
            Total number of subjects.
        blocks : np.ndarray
            Block assignment for each subject.
        n_permutations : int
            Number of permutations to generate.

        Returns
        -------
        np.ndarray
            Matrix of blocked permutations, shape (n_subjects, n_permutations).
        """
        unique_blocks = np.unique(blocks)
        permutation_matrix = np.zeros((n_subjects, n_permutations), dtype=int)

        for perm_idx in range(n_permutations):
            perm = np.zeros(n_subjects, dtype=int)

            for block_id in unique_blocks:
                block_indices = np.where(blocks == block_id)[0]
                # Permute within block
                block_perm = self.rng.permutation(block_indices)
                perm[block_indices] = block_perm

            permutation_matrix[:, perm_idx] = perm

        return permutation_matrix
