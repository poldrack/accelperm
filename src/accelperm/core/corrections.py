"""
Multiple comparison correction methods for permutation testing.

This module implements various multiple comparison correction techniques
commonly used in neuroimaging analysis, following FSL randomise compatibility.

Key correction methods:
- Family-Wise Error Rate (FWER) control using max-statistic method
- False Discovery Rate (FDR) control using Benjamini-Hochberg procedure
- Bonferroni correction for voxel-wise control
- Cluster-based correction for spatial extent and mass
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.ndimage import label as scipy_label


@dataclass
class CorrectionResult:
    """Results from multiple comparison correction.

    Parameters
    ----------
    p_values : np.ndarray
        Original uncorrected p-values
    corrected_p_values : np.ndarray
        Corrected p-values after multiple comparison adjustment
    threshold : float
        Statistical threshold used for significance determination
    significant_mask : np.ndarray
        Boolean mask indicating significant voxels/tests
    method : str
        Name of correction method used
    n_comparisons : int
        Number of multiple comparisons performed
    cluster_info : Optional[Dict[str, Any]]
        Additional information for cluster-based corrections
    """

    p_values: np.ndarray
    corrected_p_values: np.ndarray
    threshold: float
    significant_mask: np.ndarray
    method: str
    n_comparisons: int
    cluster_info: dict[str, Any] | None = None


class CorrectionMethod(ABC):
    """Abstract base class for multiple comparison correction methods."""

    @abstractmethod
    def correct(
        self, data: np.ndarray, alpha: float = 0.05, **kwargs
    ) -> CorrectionResult:
        """Apply multiple comparison correction.

        Parameters
        ----------
        data : np.ndarray
            Statistical data to correct (p-values or test statistics)
        alpha : float, default=0.05
            Significance level for correction
        **kwargs
            Method-specific parameters

        Returns
        -------
        CorrectionResult
            Results of correction including adjusted p-values and significance mask
        """
        pass

    def _validate_p_values(self, p_values: np.ndarray) -> None:
        """Validate p-values are in valid range [0,1]."""
        if len(p_values) == 0:
            raise ValueError("Empty p-values array provided")

        if np.any(np.isnan(p_values)):
            raise ValueError("NaN values found in p-values")

        if np.any(np.isinf(p_values)):
            raise ValueError("Infinite values found in p-values")

        if np.any((p_values < 0) | (p_values > 1)):
            raise ValueError("P-values must be in range [0,1]")

    def _validate_alpha(self, alpha: float) -> None:
        """Validate alpha is in valid range (0,1)."""
        if not (0 < alpha < 1):
            raise ValueError(f"Alpha must be in range (0,1), got {alpha}")


class BonferroniCorrection(CorrectionMethod):
    """Bonferroni correction for multiple comparisons.

    The most conservative correction method that controls Family-Wise Error Rate
    by adjusting the significance threshold to alpha/n where n is the number of tests.
    """

    def correct(self, p_values: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
        """Apply Bonferroni correction.

        Parameters
        ----------
        p_values : np.ndarray
            Uncorrected p-values
        alpha : float, default=0.05
            Family-wise error rate to control

        Returns
        -------
        CorrectionResult
            Bonferroni-corrected results
        """
        self._validate_p_values(p_values)
        self._validate_alpha(alpha)

        n_comparisons = len(p_values)

        # Bonferroni adjusted p-values: p_adj = min(1.0, p * n)
        corrected_p = np.minimum(1.0, p_values * n_comparisons)

        # Bonferroni threshold: alpha / n
        threshold = alpha / n_comparisons

        # Significance based on original p-values vs threshold
        significant_mask = p_values <= threshold

        return CorrectionResult(
            p_values=p_values,
            corrected_p_values=corrected_p,
            threshold=threshold,
            significant_mask=significant_mask,
            method="bonferroni",
            n_comparisons=n_comparisons,
        )


class FDRCorrection(CorrectionMethod):
    """False Discovery Rate correction using Benjamini-Hochberg procedure.

    Controls the expected proportion of false discoveries among rejected hypotheses.
    Less conservative than FWER methods, allowing for more statistical power.
    """

    def __init__(self, conservative: bool = False):
        """Initialize FDR correction.

        Parameters
        ----------
        conservative : bool, default=False
            Whether to use conservative correction factor for dependent tests
        """
        self.conservative = conservative

    def correct(self, p_values: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
        """Apply Benjamini-Hochberg FDR correction.

        Parameters
        ----------
        p_values : np.ndarray
            Uncorrected p-values
        alpha : float, default=0.05
            False discovery rate to control

        Returns
        -------
        CorrectionResult
            FDR-corrected results
        """
        self._validate_p_values(p_values)
        self._validate_alpha(alpha)

        n_comparisons = len(p_values)

        if n_comparisons == 1:
            # No correction needed for single test
            return CorrectionResult(
                p_values=p_values,
                corrected_p_values=p_values,
                threshold=alpha,
                significant_mask=p_values <= alpha,
                method="fdr_bh",
                n_comparisons=n_comparisons,
            )

        # Calculate correction factor for dependent tests if requested
        correction_factor = 1.0
        if self.conservative:
            correction_factor = np.sum(1.0 / np.arange(1, n_comparisons + 1))

        # Sort p-values with original indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]

        # Calculate ranks (1-based)
        ranks = np.arange(1, n_comparisons + 1)

        # Find largest k where p(k) <= (k/m) * alpha / correction_factor
        thresholds = (ranks / n_comparisons) * alpha / correction_factor
        significant_sorted = sorted_p_values <= thresholds

        # Find the largest significant index (step-up procedure)
        significant_indices = np.where(significant_sorted)[0]
        if len(significant_indices) > 0:
            max_significant_rank = significant_indices[-1] + 1  # +1 for 1-based rank
        else:
            max_significant_rank = 0

        # Create significance mask
        significant_mask = np.zeros(n_comparisons, dtype=bool)
        if max_significant_rank > 0:
            # All p-values up to max_significant_rank are significant
            significant_sorted_mask = np.zeros(n_comparisons, dtype=bool)
            significant_sorted_mask[:max_significant_rank] = True
            significant_mask[sorted_indices] = significant_sorted_mask

        # Calculate adjusted p-values (Benjamini-Hochberg step-down)
        adjusted_p_values = np.zeros(n_comparisons)

        # Work backwards through sorted p-values
        previous_adjusted = 1.0
        for i in range(n_comparisons - 1, -1, -1):
            rank = i + 1  # 1-based rank
            original_idx = sorted_indices[i]

            # BH adjustment: p_adj = min(previous_adj, p * m / rank * correction_factor)
            current_adjusted = min(
                previous_adjusted,
                sorted_p_values[i] * n_comparisons / rank * correction_factor,
            )
            adjusted_p_values[original_idx] = current_adjusted
            previous_adjusted = current_adjusted

        # Threshold is the largest p-value that was deemed significant
        threshold = 0.0
        if max_significant_rank > 0:
            threshold = sorted_p_values[max_significant_rank - 1]

        return CorrectionResult(
            p_values=p_values,
            corrected_p_values=adjusted_p_values,
            threshold=threshold,
            significant_mask=significant_mask,
            method="fdr_bh",
            n_comparisons=n_comparisons,
        )


class FWERCorrection(CorrectionMethod):
    """Family-Wise Error Rate correction using max-statistic method.

    Controls the probability of making one or more false discoveries across
    all tests by using the maximum statistic from the null distribution.
    """

    def __init__(self, null_distribution: np.ndarray):
        """Initialize FWER correction.

        Parameters
        ----------
        null_distribution : np.ndarray, shape (n_permutations, n_voxels)
            Null distribution from permutation testing
        """
        self.null_distribution = null_distribution
        self.method = "fwer_max_stat"

        # Calculate maximum statistic for each permutation
        self.max_null_distribution = np.max(null_distribution, axis=1)

    def correct(
        self, observed_statistics: np.ndarray, alpha: float = 0.05
    ) -> CorrectionResult:
        """Apply FWER correction using max-statistic method.

        Parameters
        ----------
        observed_statistics : np.ndarray
            Observed test statistics (not p-values)
        alpha : float, default=0.05
            Family-wise error rate to control

        Returns
        -------
        CorrectionResult
            FWER-corrected results
        """
        self._validate_alpha(alpha)

        n_comparisons = len(observed_statistics)
        n_permutations = len(self.max_null_distribution)

        # Warn if insufficient permutations for desired alpha level
        min_permutations_needed = int(1.0 / alpha)
        if n_permutations < min_permutations_needed:
            warnings.warn(
                f"Only {n_permutations} permutations available for alpha={alpha}. "
                f"Need at least {min_permutations_needed} permutations for reliable inference.",
                UserWarning,
            )

        # Calculate FWER-corrected p-values
        corrected_p_values = np.zeros(n_comparisons)

        for i, stat in enumerate(observed_statistics):
            # P-value = proportion of max null stats >= observed stat
            corrected_p_values[i] = np.mean(self.max_null_distribution >= stat)

        # Significance determination
        significant_mask = corrected_p_values <= alpha

        # Calculate threshold (95th percentile of max null distribution for reporting)
        threshold_percentile = (1 - alpha) * 100
        threshold = np.percentile(self.max_null_distribution, threshold_percentile)

        return CorrectionResult(
            p_values=corrected_p_values,  # These are already corrected p-values
            corrected_p_values=corrected_p_values,
            threshold=threshold,
            significant_mask=significant_mask,
            method="fwer_max_stat",
            n_comparisons=n_comparisons,
        )


class ClusterCorrection(CorrectionMethod):
    """Cluster-based correction for spatial extent or mass.

    Corrects for multiple comparisons by considering spatially connected
    clusters of activation rather than individual voxels.
    """

    def __init__(
        self,
        null_cluster_sizes: np.ndarray,
        voxel_threshold: float,
        connectivity: int = 26,
        correction_type: str = "extent",
    ):
        """Initialize cluster correction.

        Parameters
        ----------
        null_cluster_sizes : np.ndarray
            Null distribution of cluster sizes from permutation testing
        voxel_threshold : float
            Initial threshold for forming clusters
        connectivity : int, default=26
            Spatial connectivity type (6, 18, or 26 for 3D)
        correction_type : str, default="extent"
            Type of cluster correction ("extent" or "mass")
        """
        self.null_cluster_sizes = null_cluster_sizes
        self.voxel_threshold = voxel_threshold
        self.connectivity = connectivity
        self.correction_type = correction_type
        self.method = f"cluster_{correction_type}"

        if connectivity not in [6, 18, 26]:
            raise ValueError(f"Connectivity must be 6, 18, or 26, got {connectivity}")

    def correct(
        self,
        statistics: np.ndarray,
        alpha: float = 0.05,
        spatial_shape: tuple[int, ...] | None = None,
    ) -> CorrectionResult:
        """Apply cluster-based correction.

        Parameters
        ----------
        statistics : np.ndarray
            Statistical map (flattened)
        alpha : float, default=0.05
            Cluster-wise error rate to control
        spatial_shape : Tuple[int, ...], optional
            Original spatial shape of statistics for 3D processing

        Returns
        -------
        CorrectionResult
            Cluster-corrected results
        """
        self._validate_alpha(alpha)

        if spatial_shape is None:
            # Assume 1D processing if no shape provided
            spatial_shape = (len(statistics),)

        # Reshape statistics to spatial dimensions
        stats_spatial = statistics.reshape(spatial_shape)

        # Threshold the statistical map
        binary_map = stats_spatial > self.voxel_threshold

        # Find connected components
        if len(spatial_shape) == 3:
            # 3D connectivity structure
            if self.connectivity == 6:
                structure = ndimage.generate_binary_structure(3, 1)  # Faces only
            elif self.connectivity == 18:
                structure = ndimage.generate_binary_structure(3, 2)  # Faces + edges
            else:  # 26
                structure = ndimage.generate_binary_structure(3, 3)  # All neighbors
        else:
            # Default structure for other dimensions
            structure = None

        cluster_labels, n_clusters = scipy_label(binary_map, structure=structure)

        if n_clusters == 0:
            # No clusters found
            return CorrectionResult(
                p_values=np.ones(len(statistics)),
                corrected_p_values=np.ones(len(statistics)),
                threshold=alpha,
                significant_mask=np.zeros(len(statistics), dtype=bool),
                method=self.method,
                n_comparisons=len(statistics),
                cluster_info={
                    "cluster_sizes": np.array([]),
                    "cluster_labels": cluster_labels.flatten(),
                    "n_clusters": 0,
                },
            )

        # Calculate cluster sizes or masses
        cluster_sizes = np.zeros(n_clusters + 1)  # +1 because labels start from 1

        if self.correction_type == "extent":
            # Count voxels in each cluster
            for i in range(1, n_clusters + 1):
                cluster_sizes[i] = np.sum(cluster_labels == i)
        else:  # mass
            # Sum statistics within each cluster
            for i in range(1, n_clusters + 1):
                cluster_mask = cluster_labels == i
                cluster_sizes[i] = np.sum(stats_spatial[cluster_mask])

        cluster_sizes = cluster_sizes[1:]  # Remove index 0 (background)

        # Calculate cluster p-values
        cluster_p_values = self._calculate_cluster_p_values(cluster_sizes)

        # Determine significant clusters
        significant_clusters = cluster_p_values <= alpha

        # Create voxel-wise significance mask
        significant_mask = np.zeros_like(statistics, dtype=bool)
        corrected_p_values = np.ones_like(statistics)

        for i, (cluster_size, cluster_p, is_sig) in enumerate(
            zip(cluster_sizes, cluster_p_values, significant_clusters, strict=False)
        ):
            cluster_label = i + 1  # Labels start from 1
            cluster_mask = (cluster_labels == cluster_label).flatten()

            if is_sig:
                significant_mask[cluster_mask] = True

            # Assign cluster p-value to all voxels in cluster
            corrected_p_values[cluster_mask] = cluster_p

        # Calculate threshold (95th percentile of null distribution)
        threshold_percentile = (1 - alpha) * 100
        threshold = np.percentile(self.null_cluster_sizes, threshold_percentile)

        return CorrectionResult(
            p_values=np.ones(
                len(statistics)
            ),  # Original p-values not directly available
            corrected_p_values=corrected_p_values,
            threshold=threshold,
            significant_mask=significant_mask,
            method=self.method,
            n_comparisons=len(statistics),
            cluster_info={
                "cluster_sizes": cluster_sizes,
                "cluster_labels": cluster_labels.flatten(),
                "cluster_p_values": cluster_p_values,
                "n_clusters": n_clusters,
                "significant_clusters": significant_clusters,
            },
        )

    def _calculate_cluster_p_values(self, observed_sizes: np.ndarray) -> np.ndarray:
        """Calculate p-values for observed cluster sizes."""
        p_values = np.zeros(len(observed_sizes))

        for i, size in enumerate(observed_sizes):
            # P-value = proportion of null sizes >= observed size
            p_values[i] = np.mean(self.null_cluster_sizes >= size)

        return p_values

    def _calculate_cluster_masses(
        self, stats_spatial: np.ndarray, cluster_labels: np.ndarray, n_clusters: int
    ) -> np.ndarray:
        """Calculate cluster masses (sum of statistics within clusters)."""
        cluster_masses = np.zeros(n_clusters)

        for i in range(1, n_clusters + 1):
            cluster_mask = cluster_labels == i
            cluster_masses[i - 1] = np.sum(stats_spatial[cluster_mask])

        return cluster_masses
