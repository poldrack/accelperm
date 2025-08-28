"""GPU processing utilities for efficient null distribution computation."""

import numpy as np
from scipy.ndimage import label as scipy_label

from accelperm.core.tfce import TFCEProcessor


def process_null_distributions(
    t_stats_all: np.ndarray,
    correction_methods: list[str],
    spatial_shape: tuple[int, ...] | None = None,
    tfce_height: float = 2.0,
    tfce_extent: float = 0.5,
    tfce_connectivity: int = 26,
    cluster_threshold: float = 2.3,
) -> dict[str, np.ndarray]:
    """
    Process t-statistics from all permutations into null distributions.

    Parameters
    ----------
    t_stats_all : np.ndarray
        T-statistics for all permutations, shape (n_voxels, n_contrasts, n_permutations)
    correction_methods : list[str]
        Methods requiring null distributions
    spatial_shape : tuple, optional
        Spatial dimensions for cluster/TFCE analysis
    tfce_height : float
        TFCE height parameter
    tfce_extent : float
        TFCE extent parameter
    tfce_connectivity : int
        TFCE connectivity
    cluster_threshold : float
        Threshold for cluster formation

    Returns
    -------
    dict[str, np.ndarray]
        Null distributions for each method
    """
    n_voxels, n_contrasts, n_permutations = t_stats_all.shape
    null_distributions = {}

    # Determine what statistics we need
    need_voxel = "voxel" in correction_methods
    need_cluster = "cluster" in correction_methods
    need_tfce = "tfce" in correction_methods

    if need_voxel:
        # Max statistic for voxel-wise FWER correction
        null_max_t = np.max(
            np.abs(t_stats_all), axis=0
        )  # (n_contrasts, n_permutations)
        null_distributions["voxel"] = null_max_t
        # Also store individual statistics for compatibility
        null_distributions["max_t"] = null_max_t  # Legacy compatibility
        null_distributions["t_stats"] = t_stats_all.transpose(
            2, 0, 1
        )  # (n_permutations, n_voxels, n_contrasts)

    if need_cluster and spatial_shape is not None:
        # Cluster-based correction
        null_max_cluster = np.zeros((n_contrasts, n_permutations))

        for perm_idx in range(n_permutations):
            for c_idx in range(n_contrasts):
                t_map = t_stats_all[:, c_idx, perm_idx].reshape(spatial_shape)

                # Find clusters above threshold
                binary_map = np.abs(t_map) > cluster_threshold
                labels, n_clusters = scipy_label(binary_map)

                if n_clusters > 0:
                    cluster_sizes = [
                        np.sum(labels == i) for i in range(1, n_clusters + 1)
                    ]
                    null_max_cluster[c_idx, perm_idx] = max(cluster_sizes)

        null_distributions["cluster"] = null_max_cluster

    if need_tfce and spatial_shape is not None:
        # TFCE-based correction
        tfce_processor = TFCEProcessor(
            height_power=tfce_height,
            extent_power=tfce_extent,
            connectivity=tfce_connectivity,
        )

        null_max_tfce = np.zeros((n_contrasts, n_permutations))

        for perm_idx in range(n_permutations):
            for c_idx in range(n_contrasts):
                t_map = t_stats_all[:, c_idx, perm_idx]

                # Apply TFCE enhancement
                tfce_enhanced = tfce_processor.enhance(t_map, spatial_shape)
                null_max_tfce[c_idx, perm_idx] = np.max(tfce_enhanced)

        null_distributions["tfce"] = null_max_tfce

    return null_distributions


def chunk_permutations(n_permutations: int, max_memory_gb: float = 8.0) -> int:
    """
    Calculate optimal chunk size for permutation processing based on memory constraints.

    Parameters
    ----------
    n_permutations : int
        Total number of permutations
    max_memory_gb : float
        Maximum memory to use in GB

    Returns
    -------
    int
        Optimal chunk size
    """
    # Rough estimate: each permutation requires ~4 bytes per voxel per contrast
    # For typical neuroimaging data (~250k voxels), this is ~1MB per permutation per contrast
    bytes_per_gb = 1024**3
    max_memory_bytes = max_memory_gb * bytes_per_gb

    # Conservative estimate for chunk size
    estimated_bytes_per_perm = 250_000 * 4 * 5  # 250k voxels, 4 bytes, 5 contrasts
    chunk_size = min(int(max_memory_bytes / estimated_bytes_per_perm), n_permutations)

    return max(chunk_size, 10)  # Minimum 10 permutations per chunk
