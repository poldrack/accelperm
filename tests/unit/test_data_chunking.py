"""Tests for data chunking system - TDD RED phase."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestDataChunker:
    """Test data chunking functionality - RED phase."""
    
    def test_data_chunker_exists(self):
        """Test that DataChunker class exists - RED phase."""
        from accelperm.core.chunking import DataChunker
        
        chunker = DataChunker()
        assert chunker is not None
        
    def test_calculate_optimal_chunk_size(self):
        """Test optimal chunk size calculation - RED phase."""
        from accelperm.core.chunking import DataChunker
        
        chunker = DataChunker()
        
        # Small data should use single chunk
        small_data_shape = (1000, 20, 3)
        chunk_size = chunker.calculate_optimal_chunk_size(
            small_data_shape, available_memory_gb=16.0
        )
        assert chunk_size == 1000  # All voxels in one chunk
        
        # Large data should be chunked (use much smaller memory limit)
        large_data_shape = (250000, 100, 10)
        chunk_size = chunker.calculate_optimal_chunk_size(
            large_data_shape, available_memory_gb=0.5  # Only 0.5GB available
        )
        assert chunk_size < 250000
        assert chunk_size > 0
        
    def test_chunk_data_iterator(self):
        """Test data chunking iterator - RED phase."""
        from accelperm.core.chunking import DataChunker
        
        chunker = DataChunker()
        
        # Create test data
        Y = np.random.randn(100, 10)  # 100 voxels, 10 subjects
        X = np.random.randn(10, 3)   # 10 subjects, 3 regressors
        
        chunks = list(chunker.chunk_data(Y, X, chunk_size=30))
        
        # Should have multiple chunks
        assert len(chunks) >= 3  # ceil(100/30) = 4 chunks
        
        # Each chunk should have correct structure
        for i, (chunk_Y, chunk_X, start_idx, end_idx) in enumerate(chunks):
            assert chunk_X is X  # X matrix should be the same for all chunks
            assert isinstance(chunk_Y, np.ndarray)
            assert chunk_Y.shape[1] == Y.shape[1]  # Same number of subjects
            assert start_idx < end_idx
            assert chunk_Y.shape[0] == end_idx - start_idx
            
        # All voxels should be covered
        total_voxels = sum(end_idx - start_idx for _, _, start_idx, end_idx in chunks)
        assert total_voxels == Y.shape[0]
        
    def test_chunk_boundaries_are_correct(self):
        """Test chunk boundary handling - RED phase."""
        from accelperm.core.chunking import DataChunker
        
        chunker = DataChunker()
        
        # Test exact division
        Y = np.random.randn(90, 5)
        X = np.random.randn(5, 2)
        
        chunks = list(chunker.chunk_data(Y, X, chunk_size=30))
        
        assert len(chunks) == 3
        assert chunks[0][2] == 0 and chunks[0][3] == 30
        assert chunks[1][2] == 30 and chunks[1][3] == 60
        assert chunks[2][2] == 60 and chunks[2][3] == 90
        
        # Test inexact division
        Y = np.random.randn(95, 5)
        chunks = list(chunker.chunk_data(Y, X, chunk_size=30))
        
        assert len(chunks) == 4
        assert chunks[-1][2] == 90 and chunks[-1][3] == 95  # Last chunk smaller
        
    def test_reconstruct_results_from_chunks(self):
        """Test reconstructing results from chunked computation - RED phase."""
        from accelperm.core.chunking import DataChunker
        
        chunker = DataChunker()
        
        # Simulate chunked GLM results
        chunk_results = [
            {
                "beta": np.random.randn(30, 3),
                "t_stat": np.random.randn(30, 2),
                "p_values": np.random.randn(30, 2),
            },
            {
                "beta": np.random.randn(30, 3),
                "t_stat": np.random.randn(30, 2),
                "p_values": np.random.randn(30, 2),
            },
            {
                "beta": np.random.randn(40, 3),  # Last chunk different size
                "t_stat": np.random.randn(40, 2),
                "p_values": np.random.randn(40, 2),
            },
        ]
        
        # Reconstruct full results
        full_result = chunker.reconstruct_results(chunk_results, total_voxels=100)
        
        assert full_result["beta"].shape == (100, 3)
        assert full_result["t_stat"].shape == (100, 2)
        assert full_result["p_values"].shape == (100, 2)
        
        # Check that data is properly concatenated
        np.testing.assert_array_equal(
            full_result["beta"][:30], chunk_results[0]["beta"]
        )
        np.testing.assert_array_equal(
            full_result["beta"][30:60], chunk_results[1]["beta"]
        )
        np.testing.assert_array_equal(
            full_result["beta"][60:100], chunk_results[2]["beta"]
        )


class TestChunkedBackendWrapper:
    """Test chunked backend wrapper - RED phase."""
    
    def test_chunked_backend_wrapper_exists(self):
        """Test that ChunkedBackendWrapper exists - RED phase."""
        from accelperm.core.chunking import ChunkedBackendWrapper
        from accelperm.backends.cpu import CPUBackend
        
        base_backend = CPUBackend()
        wrapper = ChunkedBackendWrapper(base_backend)
        assert wrapper is not None
        assert wrapper.backend is base_backend
        
    def test_chunked_backend_computes_glm_in_chunks(self):
        """Test chunked GLM computation - RED phase."""
        from accelperm.core.chunking import ChunkedBackendWrapper
        from accelperm.backends.cpu import CPUBackend
        
        base_backend = CPUBackend()
        wrapper = ChunkedBackendWrapper(base_backend, chunk_size=50)
        
        # Large dataset
        Y = np.random.randn(150, 20)  # Should be split into 3 chunks
        X = np.random.randn(20, 4)
        contrasts = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        result = wrapper.compute_glm(Y, X, contrasts)
        
        # Should return properly shaped results
        assert result["beta"].shape == (150, 4)
        assert result["t_stat"].shape == (150, 2)
        assert result["p_values"].shape == (150, 2)
        
        # Results should be statistically reasonable
        assert not np.any(np.isnan(result["beta"]))
        assert not np.any(np.isnan(result["t_stat"]))
        assert not np.any(np.isnan(result["p_values"]))
        
    def test_chunked_vs_full_computation_consistency(self):
        """Test chunked computation produces same results as full - RED phase."""
        from accelperm.core.chunking import ChunkedBackendWrapper
        from accelperm.backends.cpu import CPUBackend
        
        # Use same random seed for reproducibility
        np.random.seed(42)
        Y = np.random.randn(80, 15)
        X = np.random.randn(15, 3)
        contrasts = np.array([[0, 1, 0]])
        
        # Full computation
        base_backend = CPUBackend()
        full_result = base_backend.compute_glm(Y, X, contrasts)
        
        # Chunked computation
        wrapper = ChunkedBackendWrapper(base_backend, chunk_size=25)
        chunked_result = wrapper.compute_glm(Y, X, contrasts)
        
        # Results should be nearly identical
        np.testing.assert_allclose(
            full_result["beta"], chunked_result["beta"], rtol=1e-10
        )
        np.testing.assert_allclose(
            full_result["t_stat"], chunked_result["t_stat"], rtol=1e-10
        )
        np.testing.assert_allclose(
            full_result["p_values"], chunked_result["p_values"], rtol=1e-10
        )
        
    def test_chunked_backend_handles_memory_estimation(self):
        """Test memory-based chunk size selection - RED phase."""
        from accelperm.core.chunking import ChunkedBackendWrapper
        from accelperm.backends.cpu import CPUBackend
        
        base_backend = CPUBackend()
        
        # Auto chunk size based on available memory
        wrapper = ChunkedBackendWrapper(base_backend, max_memory_gb=1.0)
        
        # Large dataset that would exceed 1GB if processed at once
        Y = np.random.randn(100000, 50).astype(np.float32)
        X = np.random.randn(50, 5).astype(np.float32)
        contrasts = np.array([[1, 0, 0, 0, 0]], dtype=np.float32)
        
        result = wrapper.compute_glm(Y, X, contrasts)
        
        # Should complete without memory issues
        assert result["beta"].shape == (100000, 5)
        assert result["t_stat"].shape == (100000, 1)
        assert result["p_values"].shape == (100000, 1)


class TestChunkingIntegration:
    """Test chunking integration with backends - RED phase."""
    
    def test_backend_factory_enables_chunking(self):
        """Test backend factory can enable chunking - RED phase."""
        from accelperm.backends.factory import BackendFactory
        
        factory = BackendFactory()
        
        # Get backend with chunking enabled
        backend = factory.get_backend("cpu", enable_chunking=True, max_memory_gb=2.0)
        
        # Should return a chunked wrapper
        assert hasattr(backend, "chunk_size") or hasattr(backend, "max_memory_gb")
        assert backend.is_available()
        
    def test_chunking_with_different_backends(self):
        """Test chunking works with different backend types - RED phase."""
        from accelperm.core.chunking import ChunkedBackendWrapper
        from accelperm.backends.cpu import CPUBackend
        
        backends_to_test = [CPUBackend()]
        
        # Add MPS backend if available
        try:
            from accelperm.backends.mps import MPSBackend
            import torch
            if torch.backends.mps.is_available():
                backends_to_test.append(MPSBackend())
        except ImportError:
            pass
            
        for base_backend in backends_to_test:
            wrapper = ChunkedBackendWrapper(base_backend, chunk_size=30)
            
            # Small test
            Y = np.random.randn(60, 10)
            X = np.random.randn(10, 2)
            contrasts = np.array([[1, 0]])
            
            result = wrapper.compute_glm(Y, X, contrasts)
            
            assert result["beta"].shape == (60, 2)
            assert result["t_stat"].shape == (60, 1)
            assert result["p_values"].shape == (60, 1)