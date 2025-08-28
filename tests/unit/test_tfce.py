"""Tests for TFCE (Threshold-Free Cluster Enhancement) - TDD RED phase."""

import numpy as np
import pytest

from accelperm.core.tfce import TFCEProcessor, connected_components_3d, tfce_enhance


class TestTFCEProcessor:
    """Test TFCE processor class - RED phase."""

    def test_tfce_processor_exists(self):
        """Test that TFCEProcessor class exists - RED phase."""
        processor = TFCEProcessor()
        assert isinstance(processor, TFCEProcessor)
        assert processor.height_power == 2.0  # Default H parameter
        assert processor.extent_power == 0.5  # Default E parameter
        assert processor.connectivity == 26  # Default connectivity
        assert processor.n_steps == 100  # Default number of steps

    def test_tfce_processor_with_custom_params(self):
        """Test TFCEProcessor with custom parameters - RED phase."""
        processor = TFCEProcessor(
            height_power=1.5,
            extent_power=0.8,
            connectivity=18,
            n_steps=50,
        )
        assert processor.height_power == 1.5
        assert processor.extent_power == 0.8
        assert processor.connectivity == 18
        assert processor.n_steps == 50

    def test_tfce_processor_invalid_connectivity(self):
        """Test TFCEProcessor rejects invalid connectivity - RED phase."""
        with pytest.raises(ValueError, match="Connectivity must be 6, 18, or 26"):
            TFCEProcessor(connectivity=10)

    def test_tfce_enhance_2d_simple(self):
        """Test TFCE enhancement on simple 2D data - RED phase."""
        processor = TFCEProcessor(connectivity=8)  # 2D connectivity

        # Simple 2D statistical map with one cluster
        stat_map = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 3.0, 0.0],
                [0.0, 2.5, 3.5, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        # Flatten for processing (as expected by the processor)
        flat_stats = stat_map.flatten()
        spatial_shape = stat_map.shape

        enhanced = processor.enhance(flat_stats, spatial_shape)

        # Enhanced values should be different from original
        assert enhanced.shape == flat_stats.shape
        assert not np.array_equal(enhanced, flat_stats)

        # Enhanced values for cluster voxels should be > 0
        cluster_mask = flat_stats > 0
        assert np.all(enhanced[cluster_mask] > 0)

        # Background should remain 0
        assert np.all(enhanced[~cluster_mask] == 0)

    def test_tfce_enhance_3d_simple(self):
        """Test TFCE enhancement on simple 3D data - RED phase."""
        processor = TFCEProcessor()

        # Simple 3D statistical map
        stat_map = np.zeros((4, 4, 3))
        stat_map[1:3, 1:3, 1] = 3.0  # Central cluster

        flat_stats = stat_map.flatten()
        spatial_shape = stat_map.shape

        enhanced = processor.enhance(flat_stats, spatial_shape)

        assert enhanced.shape == flat_stats.shape
        assert not np.array_equal(enhanced, flat_stats)

        # Cluster voxels should be enhanced
        cluster_mask = flat_stats > 0
        assert np.all(enhanced[cluster_mask] > 0)

    def test_tfce_enhance_multiple_clusters(self):
        """Test TFCE with multiple separate clusters - RED phase."""
        processor = TFCEProcessor()

        # Two separate clusters
        stat_map = np.zeros((6, 6, 3))
        stat_map[1:3, 1:3, 1] = 2.0  # First cluster
        stat_map[4:6, 4:6, 1] = 3.0  # Second cluster (higher intensity)

        flat_stats = stat_map.flatten()
        spatial_shape = stat_map.shape

        enhanced = processor.enhance(flat_stats, spatial_shape)

        # Both clusters should be enhanced
        cluster1_mask = (stat_map == 2.0).flatten()
        cluster2_mask = (stat_map == 3.0).flatten()

        assert np.all(enhanced[cluster1_mask] > 0)
        assert np.all(enhanced[cluster2_mask] > 0)

        # Higher intensity cluster should generally have higher TFCE values
        mean_enhanced_cluster1 = np.mean(enhanced[cluster1_mask])
        mean_enhanced_cluster2 = np.mean(enhanced[cluster2_mask])
        assert mean_enhanced_cluster2 > mean_enhanced_cluster1

    def test_tfce_enhance_no_positive_values(self):
        """Test TFCE with no positive values - RED phase."""
        processor = TFCEProcessor()

        # All negative or zero values
        stat_map = np.array([[-1.0, 0.0], [0.0, -2.0]])
        flat_stats = stat_map.flatten()
        spatial_shape = stat_map.shape

        enhanced = processor.enhance(flat_stats, spatial_shape)

        # Should return zeros for all negative/zero input
        assert np.all(enhanced == 0.0)

    def test_tfce_enhance_preserves_shape(self):
        """Test TFCE preserves input shape - RED phase."""
        processor = TFCEProcessor()

        # Various shapes to test
        shapes = [(10,), (5, 5), (3, 4, 5), (2, 3, 4, 5)]

        for shape in shapes:
            stat_map = np.random.randn(*shape) * 2.0  # Some positive, some negative
            flat_stats = stat_map.flatten()

            enhanced = processor.enhance(flat_stats, shape)
            assert enhanced.shape == flat_stats.shape


class TestConnectedComponents3D:
    """Test 3D connected components labeling - RED phase."""

    def test_connected_components_3d_exists(self):
        """Test connected components function exists - RED phase."""
        binary_volume = np.ones((2, 2, 2))
        labels, sizes = connected_components_3d(binary_volume, connectivity=26)

        assert isinstance(labels, np.ndarray)
        assert isinstance(sizes, np.ndarray)
        assert labels.shape == binary_volume.shape

    def test_connected_components_single_cluster(self):
        """Test connected components with single cluster - RED phase."""
        # Single connected component
        binary_volume = np.array(
            [
                [[0, 0], [0, 0]],
                [[1, 1], [1, 0]],
            ]
        )

        labels, sizes = connected_components_3d(binary_volume, connectivity=26)

        # Should find exactly one component
        unique_labels = np.unique(labels[labels > 0])
        assert len(unique_labels) == 1
        assert len(sizes) == 1
        assert sizes[0] == 3  # Three voxels in the component

    def test_connected_components_multiple_clusters(self):
        """Test connected components with multiple clusters - RED phase."""
        # Two separate components
        binary_volume = np.array(
            [
                [[1, 0, 1], [0, 0, 0]],
                [[0, 0, 0], [1, 1, 0]],
            ]
        )

        labels, sizes = connected_components_3d(binary_volume, connectivity=6)

        # Should find multiple components (depends on connectivity)
        unique_labels = np.unique(labels[labels > 0])
        assert len(unique_labels) >= 2
        assert len(sizes) >= 2

    def test_connected_components_connectivity_6(self):
        """Test 6-connectivity (faces only) - RED phase."""
        # Diagonal connection that's not connected with 6-connectivity
        binary_volume = np.array(
            [
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]],
            ]
        )

        labels, sizes = connected_components_3d(binary_volume, connectivity=6)

        # Should be two separate components with 6-connectivity
        unique_labels = np.unique(labels[labels > 0])
        assert len(unique_labels) == 2

    def test_connected_components_connectivity_26(self):
        """Test 26-connectivity (all neighbors) - RED phase."""
        # Diagonal connection that IS connected with 26-connectivity
        binary_volume = np.array(
            [
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]],
            ]
        )

        labels, sizes = connected_components_3d(binary_volume, connectivity=26)

        # Should be one component with 26-connectivity
        unique_labels = np.unique(labels[labels > 0])
        assert len(unique_labels) == 1

    def test_connected_components_empty_volume(self):
        """Test connected components with empty volume - RED phase."""
        binary_volume = np.zeros((3, 3, 3))

        labels, sizes = connected_components_3d(binary_volume, connectivity=26)

        assert np.all(labels == 0)
        assert len(sizes) == 0


class TestTFCEUtilityFunction:
    """Test standalone TFCE utility function - RED phase."""

    def test_tfce_enhance_function_exists(self):
        """Test standalone tfce_enhance function exists - RED phase."""
        stat_map = np.random.randn(20)
        spatial_shape = (4, 5)

        enhanced = tfce_enhance(stat_map, spatial_shape)

        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape == stat_map.shape

    def test_tfce_enhance_with_custom_parameters(self):
        """Test tfce_enhance with custom parameters - RED phase."""
        stat_map = np.random.randn(25) * 2
        spatial_shape = (5, 5)

        enhanced = tfce_enhance(
            stat_map,
            spatial_shape,
            height_power=1.0,
            extent_power=1.0,
            connectivity=6,
            n_steps=50,
        )

        assert enhanced.shape == stat_map.shape


class TestTFCEIntegration:
    """Test TFCE integration scenarios - RED phase."""

    def test_tfce_realistic_neuroimaging_data(self):
        """Test TFCE with realistic neuroimaging dimensions - RED phase."""
        processor = TFCEProcessor()

        # Simulate realistic neuroimaging data dimensions
        shape = (64, 64, 30)  # Typical fMRI slice dimensions
        n_voxels = np.prod(shape)

        # Create statistical map with some clusters
        stat_map = np.random.randn(n_voxels) * 0.5  # Mostly noise

        # Add some "activation" clusters
        cluster_indices = np.random.choice(n_voxels, size=100, replace=False)
        stat_map[cluster_indices] += np.random.randn(100) * 2 + 3  # Higher values

        enhanced = processor.enhance(stat_map, shape)

        assert enhanced.shape == stat_map.shape
        assert not np.array_equal(enhanced, stat_map)

        # TFCE should enhance positive values
        positive_mask = stat_map > 0
        if np.any(positive_mask):
            assert np.any(enhanced[positive_mask] > 0)

    def test_tfce_performance_benchmark(self):
        """Test TFCE performance with large data - RED phase."""
        processor = TFCEProcessor(n_steps=10)  # Fewer steps for speed

        # Large volume test
        shape = (100, 100, 50)
        stat_map = np.random.randn(np.prod(shape))

        # Add cluster to make it interesting
        stat_map[:1000] = np.abs(np.random.randn(1000)) + 2

        enhanced = processor.enhance(stat_map, shape)

        # Should complete without error
        assert enhanced.shape == stat_map.shape

    def test_tfce_numerical_stability(self):
        """Test TFCE numerical stability with extreme values - RED phase."""
        processor = TFCEProcessor()

        # Extreme values
        stat_map = np.array([0.0, 1e-10, 1e10, np.inf])
        spatial_shape = (2, 2)

        # Should handle extreme values gracefully
        enhanced = processor.enhance(stat_map, spatial_shape)

        assert enhanced.shape == stat_map.shape
        assert np.all(np.isfinite(enhanced[:-1]))  # All but inf should be finite
