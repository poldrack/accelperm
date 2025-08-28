"""
Threshold-Free Cluster Enhancement (TFCE) implementation.

This module implements TFCE following FSL randomise compatibility,
providing statistical enhancement of spatial clusters without
arbitrary threshold selection.

Key Features:
- 3D connected components labeling with configurable connectivity
- Multi-threshold processing for comprehensive enhancement
- Memory-efficient implementation for large neuroimaging datasets
- FSL randomise parameter compatibility (H=2.0, E=0.5, dh=0.1)

References:
- Smith & Nichols (2009) - Threshold-free cluster enhancement
- FSL randomise implementation pseudocode
"""

import numpy as np


class TFCEProcessor:
    """Threshold-Free Cluster Enhancement processor.

    Implements the TFCE algorithm for enhancing statistical maps
    by integrating cluster extent and height information across
    multiple threshold levels.

    Parameters
    ----------
    height_power : float, default=2.0
        Height exponent (H parameter) - emphasizes peak height
    extent_power : float, default=0.5
        Extent exponent (E parameter) - emphasizes cluster size
    connectivity : int, default=26
        Spatial connectivity (6, 18, or 26 for 3D; 4 or 8 for 2D)
    n_steps : int, default=100
        Number of threshold steps for integration
    """

    def __init__(
        self,
        height_power: float = 2.0,
        extent_power: float = 0.5,
        connectivity: int = 26,
        n_steps: int = 100,
    ):
        if connectivity not in [6, 18, 26, 4, 8]:
            raise ValueError("Connectivity must be 6, 18, or 26")

        self.height_power = height_power
        self.extent_power = extent_power
        self.connectivity = connectivity
        self.n_steps = n_steps

    def enhance(
        self, stat_map: np.ndarray, spatial_shape: tuple[int, ...]
    ) -> np.ndarray:
        """Apply TFCE enhancement to statistical map.

        Parameters
        ----------
        stat_map : np.ndarray
            Flattened statistical map to enhance
        spatial_shape : Tuple[int, ...]
            Original spatial dimensions of the statistical map

        Returns
        -------
        np.ndarray
            TFCE-enhanced statistical map (same shape as input)
        """
        # Handle no positive values
        if np.all(stat_map <= 0):
            return np.zeros_like(stat_map)

        # Reshape to spatial dimensions
        spatial_map = stat_map.reshape(spatial_shape)

        # Initialize enhanced output
        enhanced = np.zeros_like(spatial_map)

        # Find maximum value for step calculation
        max_val = np.max(stat_map[stat_map > 0])

        if max_val <= 0:
            return np.zeros_like(stat_map)

        # Calculate step size
        step_size = max_val / self.n_steps

        # Process each threshold level
        for i in range(self.n_steps):
            threshold = (i + 1) * step_size

            # Create binary mask at current threshold
            binary_mask = spatial_map >= threshold

            if not np.any(binary_mask):
                continue

            # Find connected components
            labels, sizes = connected_components_3d(binary_mask, self.connectivity)

            # Apply TFCE enhancement for this threshold
            for component_id in range(1, len(sizes) + 1):
                component_mask = labels == component_id
                if np.any(component_mask):
                    cluster_size = sizes[
                        component_id - 1
                    ]  # Convert to 0-based indexing
                    enhancement = (cluster_size**self.extent_power) * (
                        threshold**self.height_power
                    )
                    enhanced[component_mask] += enhancement * step_size

        return enhanced.flatten()


def connected_components_3d(
    binary_volume: np.ndarray, connectivity: int = 26
) -> tuple[np.ndarray, np.ndarray]:
    """Find connected components in 3D binary volume.

    Parameters
    ----------
    binary_volume : np.ndarray
        Binary volume to analyze
    connectivity : int, default=26
        Spatial connectivity type (6, 18, or 26)

    Returns
    -------
    labels : np.ndarray
        Connected component labels (same shape as input)
    sizes : np.ndarray
        Size of each connected component
    """
    from scipy.ndimage import generate_binary_structure
    from scipy.ndimage import label as scipy_label

    if np.all(binary_volume == 0):
        return np.zeros_like(binary_volume), np.array([])

    # Create connectivity structure based on actual dimensionality
    ndim = binary_volume.ndim

    if ndim == 1:
        # 1D connectivity (adjacency)
        structure = generate_binary_structure(1, 1)
    elif ndim == 2:
        # 2D connectivity
        if connectivity == 4:
            structure = generate_binary_structure(2, 1)  # 4-connectivity
        else:  # connectivity == 8
            structure = generate_binary_structure(2, 2)  # 8-connectivity
    elif ndim == 3:
        # 3D connectivity
        if connectivity == 6:
            structure = generate_binary_structure(3, 1)  # Faces only
        elif connectivity == 18:
            structure = generate_binary_structure(3, 2)  # Faces + edges
        else:  # connectivity == 26
            structure = generate_binary_structure(3, 3)  # All neighbors
    else:
        # Higher dimensions - use full connectivity
        structure = generate_binary_structure(ndim, ndim)

    # Find connected components
    labels, n_features = scipy_label(binary_volume, structure=structure)

    if n_features == 0:
        return labels, np.array([])

    # Calculate cluster sizes
    sizes = np.zeros(n_features)
    for i in range(1, n_features + 1):
        sizes[i - 1] = np.sum(labels == i)

    return labels, sizes


def tfce_enhance(
    stat_map: np.ndarray,
    spatial_shape: tuple[int, ...],
    height_power: float = 2.0,
    extent_power: float = 0.5,
    connectivity: int = 26,
    n_steps: int = 100,
) -> np.ndarray:
    """Standalone TFCE enhancement function.

    Parameters
    ----------
    stat_map : np.ndarray
        Flattened statistical map
    spatial_shape : Tuple[int, ...]
        Original spatial shape
    height_power : float, default=2.0
        Height exponent (H parameter)
    extent_power : float, default=0.5
        Extent exponent (E parameter)
    connectivity : int, default=26
        Spatial connectivity
    n_steps : int, default=100
        Number of threshold steps

    Returns
    -------
    np.ndarray
        TFCE-enhanced statistical map
    """
    processor = TFCEProcessor(
        height_power=height_power,
        extent_power=extent_power,
        connectivity=connectivity,
        n_steps=n_steps,
    )
    return processor.enhance(stat_map, spatial_shape)
