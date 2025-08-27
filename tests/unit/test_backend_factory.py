"""Tests for backend factory implementation - TDD RED phase."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from accelperm.backends.base import Backend


class TestBackendFactory:
    """Test backend factory functionality - RED phase."""
    
    def test_backend_factory_exists(self):
        """Test that BackendFactory class exists - RED phase."""
        from accelperm.backends.factory import BackendFactory
        
        factory = BackendFactory()
        assert factory is not None
        
    def test_backend_factory_auto_detect_best_available(self):
        """Test automatic best backend detection - RED phase."""
        from accelperm.backends.factory import BackendFactory
        
        factory = BackendFactory()
        
        # Should detect the best available backend
        backend = factory.get_best_backend()
        assert backend is not None
        assert backend.is_available()
        
        # Should prefer MPS over CPU if available
        available_backends = factory.list_available_backends()
        if "mps" in available_backends:
            assert backend.name == "mps"
        else:
            assert backend.name == "cpu"
        
    def test_backend_factory_prefers_mps_when_available(self):
        """Test MPS backend preference when available - RED phase."""
        from accelperm.backends.factory import BackendFactory
        
        factory = BackendFactory()
        
        with patch('torch.backends.mps.is_available', return_value=True):
            backend = factory.get_best_backend()
            assert backend.name == "mps"
            assert backend.is_available()
            
    def test_backend_factory_fallback_to_cpu_when_mps_unavailable(self):
        """Test fallback to CPU when MPS unavailable - RED phase."""
        from accelperm.backends.factory import BackendFactory
        
        factory = BackendFactory()
        
        with patch('torch.backends.mps.is_available', return_value=False):
            backend = factory.get_best_backend()
            assert backend.name == "cpu"
            assert backend.is_available()
            
    def test_backend_factory_respects_user_preference(self):
        """Test user backend preference override - RED phase."""
        from accelperm.backends.factory import BackendFactory
        
        factory = BackendFactory()
        
        # User explicitly requests CPU
        backend = factory.get_backend("cpu")
        assert backend.name == "cpu"
        
        # User explicitly requests MPS (if available)
        with patch('torch.backends.mps.is_available', return_value=True):
            backend = factory.get_backend("mps")
            assert backend.name == "mps"
            
    def test_backend_factory_validates_user_preference(self):
        """Test validation of user backend preference - RED phase."""
        from accelperm.backends.factory import BackendFactory
        
        factory = BackendFactory()
        
        # Invalid backend should raise error
        with pytest.raises(ValueError, match="Invalid backend"):
            factory.get_backend("invalid")
            
        # Unavailable backend should raise error
        with patch('torch.backends.mps.is_available', return_value=False):
            with pytest.raises(RuntimeError, match="not available"):
                factory.get_backend("mps")


class TestBackendSelection:
    """Test intelligent backend selection - RED phase."""
    
    def test_memory_requirement_estimation(self):
        """Test memory requirement estimation - RED phase."""
        from accelperm.backends.factory import BackendFactory
        
        factory = BackendFactory()
        
        # Small dataset should work on any backend
        data_shape = (1000, 20, 3)  # voxels, subjects, regressors
        memory_mb = factory.estimate_memory_requirements(data_shape)
        
        assert memory_mb > 0
        assert isinstance(memory_mb, (int, float))
        
    def test_backend_selection_based_on_data_size(self):
        """Test backend selection based on data size - RED phase."""
        from accelperm.backends.factory import BackendFactory
        
        factory = BackendFactory()
        
        # Small dataset - should prefer CPU for low overhead
        small_data = (100, 10, 2)
        small_backend = factory.get_optimal_backend(small_data)
        
        # Large dataset - should prefer GPU if available
        large_data = (100000, 100, 10)
        with patch('torch.backends.mps.is_available', return_value=True):
            large_backend = factory.get_optimal_backend(large_data)
            
        # Both should return valid backends
        assert small_backend is not None
        assert large_backend is not None
        assert isinstance(small_backend, Backend)
        assert isinstance(large_backend, Backend)
        
    def test_available_backends_listing(self):
        """Test listing of available backends - RED phase."""
        from accelperm.backends.factory import BackendFactory
        
        factory = BackendFactory()
        
        available = factory.list_available_backends()
        
        # Should always include CPU
        assert "cpu" in available
        assert isinstance(available, list)
        
        # Should include MPS if available
        with patch('torch.backends.mps.is_available', return_value=True):
            available_with_mps = factory.list_available_backends()
            assert "mps" in available_with_mps
            
    def test_backend_capabilities_query(self):
        """Test querying backend capabilities - RED phase."""
        from accelperm.backends.factory import BackendFactory
        
        factory = BackendFactory()
        
        cpu_caps = factory.get_backend_capabilities("cpu")
        assert isinstance(cpu_caps, dict)
        assert "max_memory_gb" in cpu_caps
        assert "supports_float64" in cpu_caps
        
        with patch('torch.backends.mps.is_available', return_value=True):
            mps_caps = factory.get_backend_capabilities("mps")
            assert isinstance(mps_caps, dict)
            assert "max_memory_gb" in mps_caps
            assert "supports_float64" in mps_caps


class TestBackendIntegration:
    """Test backend factory integration - RED phase."""
    
    def test_factory_creates_working_backends(self):
        """Test that factory creates working backends - RED phase."""
        from accelperm.backends.factory import BackendFactory
        
        factory = BackendFactory()
        
        # Test CPU backend creation and basic functionality
        cpu_backend = factory.get_backend("cpu")
        assert cpu_backend.is_available()
        
        # Simple GLM test
        Y = np.array([[1.0, 2.0], [3.0, 4.0]])
        X = np.array([[1.0, 0.0], [1.0, 1.0]])
        contrasts = np.array([[0.0, 1.0]])
        
        result = cpu_backend.compute_glm(Y, X, contrasts)
        assert "beta" in result
        assert "t_stat" in result
        assert "p_values" in result
        
    def test_factory_caches_backend_instances(self):
        """Test backend instance caching - RED phase."""
        from accelperm.backends.factory import BackendFactory
        
        factory = BackendFactory()
        
        # Should return same instance for repeated calls
        backend1 = factory.get_backend("cpu")
        backend2 = factory.get_backend("cpu")
        
        assert backend1 is backend2
        
    def test_factory_thread_safety(self):
        """Test factory thread safety - RED phase."""
        import threading
        from accelperm.backends.factory import BackendFactory
        
        factory = BackendFactory()
        backends = []
        
        def get_backend():
            backend = factory.get_backend("cpu")
            backends.append(backend)
            
        # Create multiple threads
        threads = [threading.Thread(target=get_backend) for _ in range(10)]
        
        # Start all threads
        for t in threads:
            t.start()
            
        # Wait for all threads
        for t in threads:
            t.join()
            
        # All should return the same instance (cached)
        assert len(set(id(b) for b in backends)) == 1