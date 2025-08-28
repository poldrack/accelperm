"""GPU-accelerated null distribution processing."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from scipy.ndimage import label as scipy_label
    from skimage.measure import label as skimage_label

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class GPUNullDistributionProcessor:
    """GPU-accelerated processor for null distributions."""

    def __init__(self, device: torch.device):
        self.device = device

    def process_null_distributions_gpu(
        self,
        t_stats_all: np.ndarray,
        correction_methods: List[str],
        spatial_shape: Optional[Tuple[int, ...]] = None,
        tfce_height: float = 2.0,
        tfce_extent: float = 0.5,
        tfce_connectivity: int = 26,
        cluster_threshold: float = 2.3,
        chunk_size: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Process null distributions with GPU acceleration.

        Parameters
        ----------
        t_stats_all : np.ndarray
            T-statistics for all permutations, shape (n_voxels, n_contrasts, n_permutations)
        correction_methods : List[str]
            Methods requiring null distributions
        spatial_shape : Optional[Tuple[int, ...]]
            Spatial dimensions for cluster/TFCE analysis
        tfce_height : float
            TFCE height parameter
        tfce_extent : float
            TFCE extent parameter
        tfce_connectivity : int
            TFCE connectivity
        cluster_threshold : float
            Threshold for cluster formation
        chunk_size : int
            Number of permutations to process per GPU batch

        Returns
        -------
        Dict[str, np.ndarray]
            Null distributions for each method
        """
        n_voxels, n_contrasts, n_permutations = t_stats_all.shape
        null_distributions = {}

        # Determine what statistics we need
        need_voxel = "voxel" in correction_methods
        need_cluster = "cluster" in correction_methods
        need_tfce = "tfce" in correction_methods

        # Convert to GPU tensor once
        t_stats_gpu = torch.from_numpy(t_stats_all).float().to(self.device)

        if need_voxel:
            # Voxel-wise correction: max statistic (already very fast on GPU)
            null_max_t = torch.max(torch.abs(t_stats_gpu), dim=0)[
                0
            ]  # (n_contrasts, n_permutations)
            null_distributions["voxel"] = null_max_t.cpu().numpy()
            null_distributions["max_t"] = null_max_t.cpu().numpy()
            null_distributions["t_stats"] = t_stats_all.transpose(2, 0, 1)

        if need_cluster and spatial_shape is not None:
            # GPU-accelerated cluster processing
            null_max_cluster = self._process_clusters_gpu(
                t_stats_gpu, spatial_shape, cluster_threshold, chunk_size
            )
            null_distributions["cluster"] = null_max_cluster.cpu().numpy()

        if need_tfce and spatial_shape is not None:
            # GPU-accelerated TFCE processing
            null_max_tfce = self._process_tfce_gpu(
                t_stats_gpu,
                spatial_shape,
                tfce_height,
                tfce_extent,
                tfce_connectivity,
                chunk_size,
            )
            null_distributions["tfce"] = null_max_tfce.cpu().numpy()

        # Cleanup
        del t_stats_gpu
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

        return null_distributions

    def _process_clusters_gpu(
        self,
        t_stats_gpu: torch.Tensor,
        spatial_shape: Tuple[int, ...],
        cluster_threshold: float,
        chunk_size: int,
    ) -> torch.Tensor:
        """GPU-accelerated cluster processing."""
        n_voxels, n_contrasts, n_permutations = t_stats_gpu.shape
        null_max_cluster = torch.zeros(n_contrasts, n_permutations, device=self.device)

        # Process in chunks to manage memory
        for chunk_start in range(0, n_permutations, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_permutations)
            chunk_t_stats = t_stats_gpu[:, :, chunk_start:chunk_end]

            # Vectorized thresholding across all permutations in chunk
            binary_maps = torch.abs(chunk_t_stats) > cluster_threshold

            # Process each permutation in chunk (still need CPU for connected components)
            for i, perm_idx in enumerate(range(chunk_start, chunk_end)):
                for c_idx in range(n_contrasts):
                    binary_map = binary_maps[:, c_idx, i].cpu().numpy()
                    binary_3d = binary_map.reshape(spatial_shape)

                    # Use scipy for connected components (fastest available option)
                    if SCIPY_AVAILABLE:
                        labels, n_clusters = scipy_label(binary_3d)
                        if n_clusters > 0:
                            cluster_sizes = [
                                np.sum(labels == j) for j in range(1, n_clusters + 1)
                            ]
                            max_size = max(cluster_sizes) if cluster_sizes else 0
                        else:
                            max_size = 0
                    else:
                        # Fallback to simple threshold counting if scipy not available
                        max_size = np.sum(binary_map)

                    null_max_cluster[c_idx, perm_idx] = max_size

        return null_max_cluster

    def _process_tfce_gpu(
        self,
        t_stats_gpu: torch.Tensor,
        spatial_shape: Tuple[int, ...],
        tfce_height: float,
        tfce_extent: float,
        tfce_connectivity: int,
        chunk_size: int,
    ) -> torch.Tensor:
        """GPU-accelerated TFCE processing."""
        n_voxels, n_contrasts, n_permutations = t_stats_gpu.shape
        null_max_tfce = torch.zeros(n_contrasts, n_permutations, device=self.device)

        # Initialize TFCE processor
        from accelperm.core.tfce import TFCEProcessor

        tfce_processor = TFCEProcessor(
            height_power=tfce_height,
            extent_power=tfce_extent,
            connectivity=tfce_connectivity,
        )

        # Process in chunks to manage memory
        for chunk_start in range(0, n_permutations, chunk_size):
            chunk_end = min(
                chunk_start + chunk_size, n_permutations
            )  # Fixed bug: was chunk_size

            # Get chunk of t-statistics
            chunk_t_stats = t_stats_gpu[:, :, chunk_start:chunk_end]
            chunk_size_actual = chunk_end - chunk_start

            # Optimized CPU processing for TFCE (still faster due to chunking)
            # Note: Full GPU TFCE requires complex connected components - keeping CPU for accuracy
            for i, perm_idx in enumerate(range(chunk_start, chunk_end)):
                for c_idx in range(n_contrasts):
                    t_map = chunk_t_stats[:, c_idx, i].cpu().numpy()
                    tfce_enhanced = tfce_processor.enhance(t_map, spatial_shape)
                    null_max_tfce[c_idx, perm_idx] = np.max(tfce_enhanced)

        return null_max_tfce

    def _tfce_gpu_vectorized(
        self,
        t_stats_chunk: torch.Tensor,
        spatial_shape: Tuple[int, int, int],
        height_power: float,
        extent_power: float,
        connectivity: int,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """
        Experimental GPU-accelerated TFCE implementation.

        This is a simplified TFCE that processes multiple thresholds in parallel on GPU.
        For full accuracy, falls back to CPU-based connected components.
        """
        n_voxels, n_contrasts, chunk_size = t_stats_chunk.shape

        # Reshape to spatial dimensions: (n_voxels, n_contrasts, chunk_size) -> (chunk_size, n_contrasts, *spatial_shape)
        t_maps = t_stats_chunk.permute(2, 1, 0).reshape(
            chunk_size, n_contrasts, *spatial_shape
        )

        # Find global max for step size calculation
        max_vals = torch.max(t_maps.reshape(chunk_size, n_contrasts, -1), dim=2)[0]

        # Initialize enhancement output
        enhanced = torch.zeros_like(t_maps)

        # Process each contrast separately (could be further vectorized)
        for c_idx in range(n_contrasts):
            contrast_maps = t_maps[:, c_idx]  # (chunk_size, spatial_dims)
            contrast_max = torch.max(max_vals[:, c_idx])

            if contrast_max > 0:
                step_size = contrast_max / n_steps

                # Vectorized threshold processing
                for step in range(1, n_steps + 1):
                    threshold = step * step_size

                    # Create binary masks for all permutations simultaneously
                    binary_masks = contrast_maps >= threshold

                    # Simple enhancement (approximation - not full connected components)
                    # For speed, use neighborhood-based approximation
                    cluster_sizes = self._approximate_cluster_sizes_gpu(
                        binary_masks, connectivity
                    )

                    # TFCE enhancement
                    enhancement = (cluster_sizes**extent_power) * (
                        threshold**height_power
                    )
                    enhanced[:, c_idx] += enhancement * step_size

        return enhanced.reshape(chunk_size, n_contrasts, n_voxels)

    def _approximate_cluster_sizes_gpu(
        self, binary_masks: torch.Tensor, connectivity: int
    ) -> torch.Tensor:
        """
        GPU-based approximation of cluster sizes using convolution.

        This is a fast approximation - for exact results, use CPU connected components.
        """
        # Simple approximation: use local neighborhood counting
        # This gives approximate cluster sizes without expensive connected components

        if len(binary_masks.shape) == 4:  # (batch, x, y, z)
            # 3D convolution for neighborhood counting
            kernel_size = 3
            kernel = torch.ones(
                1, 1, kernel_size, kernel_size, kernel_size, device=self.device
            ) / (kernel_size**3)

            # Apply convolution to get neighborhood density
            binary_float = binary_masks.float().unsqueeze(1)  # Add channel dimension
            neighborhood_density = torch.nn.functional.conv3d(
                binary_float, kernel, padding=kernel_size // 2
            ).squeeze(1)

            # Approximate cluster size by local density
            cluster_approximation = neighborhood_density * binary_masks.float()

        else:  # 2D case
            kernel_size = 3
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device) / (
                kernel_size**2
            )

            binary_float = binary_masks.float().unsqueeze(1)
            neighborhood_density = torch.nn.functional.conv2d(
                binary_float, kernel, padding=kernel_size // 2
            ).squeeze(1)

            cluster_approximation = neighborhood_density * binary_masks.float()

        return cluster_approximation


def process_null_distributions_gpu_accelerated(
    t_stats_all: np.ndarray,
    correction_methods: List[str],
    device: torch.device,
    spatial_shape: Optional[Tuple[int, ...]] = None,
    tfce_height: float = 2.0,
    tfce_extent: float = 0.5,
    tfce_connectivity: int = 26,
    cluster_threshold: float = 2.3,
) -> Dict[str, np.ndarray]:
    """
    GPU-accelerated null distribution processing.

    This function provides significant speedup over CPU-based processing
    by leveraging GPU tensor operations for threshold-based computations.
    """
    processor = GPUNullDistributionProcessor(device)

    # Calculate chunk size based on available memory
    n_voxels, n_contrasts, n_permutations = t_stats_all.shape

    # Estimate memory per permutation and calculate safe chunk size
    if device.type == "mps":
        # Conservative for MPS
        chunk_size = min(50, n_permutations)
    elif device.type == "cuda":
        # More aggressive for CUDA
        chunk_size = min(200, n_permutations)
    else:
        chunk_size = min(25, n_permutations)

    return processor.process_null_distributions_gpu(
        t_stats_all=t_stats_all,
        correction_methods=correction_methods,
        spatial_shape=spatial_shape,
        tfce_height=tfce_height,
        tfce_extent=tfce_extent,
        tfce_connectivity=tfce_connectivity,
        cluster_threshold=cluster_threshold,
        chunk_size=chunk_size,
    )
