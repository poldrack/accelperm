"""Tests for NIfTI file I/O operations - TDD RED phase."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from accelperm.io.nifti import (
    NiftiLoader,
    load_nifti,
    save_nifti,
    validate_nifti_compatibility,
)


class TestNiftiLoader:
    """Test the NiftiLoader class - RED phase."""

    def test_nifti_loader_exists(self):
        """Test that NiftiLoader class exists - RED phase."""
        # This should fail because NiftiLoader doesn't exist yet
        loader = NiftiLoader()
        assert isinstance(loader, NiftiLoader)

    def test_load_single_nifti_file(self):
        """Test loading a single NIfTI file - RED phase."""
        # This should fail because load method doesn't exist yet
        loader = NiftiLoader()
        
        # Mock file path
        test_path = Path("/fake/path/test.nii.gz")
        
        with patch("nibabel.load") as mock_nib_load:
            # Mock nibabel image object
            mock_img = Mock()
            mock_img.get_fdata.return_value = np.random.rand(64, 64, 30)
            mock_img.affine = np.eye(4)
            mock_img.header = Mock()
            mock_nib_load.return_value = mock_img
            
            result = loader.load(test_path)
            
            assert "data" in result
            assert "affine" in result
            assert "header" in result
            assert result["data"].shape == (64, 64, 30)
            assert result["affine"].shape == (4, 4)
            mock_nib_load.assert_called_once_with(str(test_path))

    def test_load_4d_nifti_file(self):
        """Test loading 4D NIfTI file with time dimension - RED phase."""
        # This should fail because 4D handling doesn't exist yet
        loader = NiftiLoader()
        
        test_path = Path("/fake/path/test_4d.nii.gz")
        
        with patch("nibabel.load") as mock_nib_load:
            mock_img = Mock()
            mock_img.get_fdata.return_value = np.random.rand(64, 64, 30, 100)
            mock_img.affine = np.eye(4)
            mock_img.header = Mock()
            mock_nib_load.return_value = mock_img
            
            result = loader.load(test_path)
            
            assert result["data"].shape == (64, 64, 30, 100)
            assert result["n_volumes"] == 100
            assert result["spatial_shape"] == (64, 64, 30)

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises appropriate error - RED phase."""
        # This should fail because error handling doesn't exist yet
        loader = NiftiLoader()
        
        nonexistent_path = Path("/fake/nonexistent.nii.gz")
        
        with pytest.raises(FileNotFoundError, match="NIfTI file not found"):
            loader.load(nonexistent_path)

    def test_load_invalid_nifti_raises_error(self):
        """Test that invalid NIfTI file raises appropriate error - RED phase."""
        # This should fail because validation doesn't exist yet
        loader = NiftiLoader()
        
        test_path = Path("/fake/path/invalid.nii.gz")
        
        with patch("nibabel.load", side_effect=Exception("Invalid NIfTI")):
            with pytest.raises(ValueError, match="Invalid NIfTI file"):
                loader.load(test_path)

    def test_load_with_mask(self):
        """Test loading NIfTI with brain mask - RED phase."""
        # This should fail because mask support doesn't exist yet
        loader = NiftiLoader()
        
        data_path = Path("/fake/path/data.nii.gz")
        mask_path = Path("/fake/path/mask.nii.gz")
        
        with patch("nibabel.load") as mock_nib_load:
            # Mock data image
            mock_data_img = Mock()
            mock_data_img.get_fdata.return_value = np.random.rand(64, 64, 30, 100)
            mock_data_img.affine = np.eye(4)
            mock_data_img.header = Mock()
            
            # Mock mask image
            mock_mask_img = Mock()
            mask_data = np.random.rand(64, 64, 30) > 0.5  # Binary mask
            mock_mask_img.get_fdata.return_value = mask_data.astype(float)
            mock_mask_img.affine = np.eye(4)
            mock_mask_img.header = Mock()
            
            def side_effect(path):
                if "mask" in str(path):
                    return mock_mask_img
                return mock_data_img
            
            mock_nib_load.side_effect = side_effect
            
            result = loader.load(data_path, mask=mask_path)
            
            assert "data" in result
            assert "mask" in result
            assert "masked_data" in result
            assert result["mask"].dtype == bool
            assert result["n_voxels"] == np.sum(mask_data)


class TestNiftiUtilityFunctions:
    """Test utility functions for NIfTI operations - RED phase."""

    def test_load_nifti_function_exists(self):
        """Test that load_nifti function exists - RED phase."""
        # This should fail because function doesn't exist yet
        test_path = Path("/fake/path/test.nii.gz")
        
        with patch("nibabel.load") as mock_nib_load:
            mock_img = Mock()
            mock_img.get_fdata.return_value = np.random.rand(64, 64, 30)
            mock_img.affine = np.eye(4)
            mock_img.header = Mock()
            mock_nib_load.return_value = mock_img
            
            data, affine, header = load_nifti(test_path)
            
            assert data.shape == (64, 64, 30)
            assert affine.shape == (4, 4)
            assert header is not None

    def test_save_nifti_function_exists(self):
        """Test that save_nifti function exists - RED phase."""
        # This should fail because function doesn't exist yet
        data = np.random.rand(64, 64, 30)
        affine = np.eye(4)
        output_path = Path("/fake/output/test.nii.gz")
        
        with patch("nibabel.Nifti1Image") as mock_nifti_img, \
             patch("pathlib.Path.mkdir") as mock_mkdir:
            
            mock_img_instance = Mock()
            mock_nifti_img.return_value = mock_img_instance
            
            save_nifti(data, affine, output_path)
            
            mock_nifti_img.assert_called_once_with(data, affine)
            mock_img_instance.to_filename.assert_called_once_with(str(output_path))
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_validate_nifti_compatibility_function_exists(self):
        """Test NIfTI compatibility validation function - RED phase."""
        # This should fail because validation doesn't exist yet
        
        # Compatible images (same spatial dimensions)
        img1_info = {"spatial_shape": (64, 64, 30), "affine": np.eye(4)}
        img2_info = {"spatial_shape": (64, 64, 30), "affine": np.eye(4)}
        
        is_compatible = validate_nifti_compatibility(img1_info, img2_info)
        assert is_compatible is True
        
        # Incompatible images (different spatial dimensions)
        img3_info = {"spatial_shape": (32, 32, 20), "affine": np.eye(4)}
        
        is_compatible = validate_nifti_compatibility(img1_info, img3_info)
        assert is_compatible is False


class TestNiftiDataValidation:
    """Test data validation and quality checks - RED phase."""

    def test_check_data_integrity(self):
        """Test data integrity checking - RED phase."""
        # This should fail because integrity checking doesn't exist yet
        loader = NiftiLoader()
        
        # Test with NaN values
        data_with_nans = np.random.rand(10, 10, 10)
        data_with_nans[5, 5, 5] = np.nan
        
        with pytest.raises(ValueError, match="Data contains NaN values"):
            loader._check_data_integrity(data_with_nans)
        
        # Test with infinite values
        data_with_inf = np.random.rand(10, 10, 10)
        data_with_inf[3, 3, 3] = np.inf
        
        with pytest.raises(ValueError, match="Data contains infinite values"):
            loader._check_data_integrity(data_with_inf)
        
        # Test with valid data
        valid_data = np.random.rand(10, 10, 10)
        # Should not raise any exception
        loader._check_data_integrity(valid_data)

    def test_validate_dimensions(self):
        """Test dimension validation - RED phase."""
        # This should fail because dimension validation doesn't exist yet
        loader = NiftiLoader()
        
        # Valid 3D data
        data_3d = np.random.rand(64, 64, 30)
        loader._validate_dimensions(data_3d, expected_ndim=3)
        
        # Valid 4D data
        data_4d = np.random.rand(64, 64, 30, 100)
        loader._validate_dimensions(data_4d, expected_ndim=4)
        
        # Invalid dimensions
        data_2d = np.random.rand(64, 64)
        with pytest.raises(ValueError, match="Expected 3D data"):
            loader._validate_dimensions(data_2d, expected_ndim=3)

    def test_validate_spatial_match(self):
        """Test spatial dimension matching validation - RED phase."""
        # This should fail because spatial validation doesn't exist yet
        loader = NiftiLoader()
        
        # Matching spatial dimensions
        data1 = np.random.rand(64, 64, 30, 50)
        data2 = np.random.rand(64, 64, 30, 75)
        
        loader._validate_spatial_match(data1, data2)  # Should not raise
        
        # Non-matching spatial dimensions
        data3 = np.random.rand(32, 32, 20, 50)
        
        with pytest.raises(ValueError, match="Spatial dimensions do not match"):
            loader._validate_spatial_match(data1, data3)


class TestNiftiMemoryOptimization:
    """Test memory-efficient NIfTI loading - RED phase."""

    def test_lazy_loading_support(self):
        """Test lazy loading for large files - RED phase."""
        # This should fail because lazy loading doesn't exist yet
        loader = NiftiLoader(lazy_loading=True)
        
        test_path = Path("/fake/path/large_file.nii.gz")
        
        with patch("nibabel.load") as mock_nib_load:
            mock_img = Mock()
            # Don't call get_fdata() immediately for lazy loading
            mock_img.get_fdata.return_value = np.random.rand(128, 128, 64, 500)
            mock_img.affine = np.eye(4)
            mock_img.header = Mock()
            mock_nib_load.return_value = mock_img
            
            result = loader.load(test_path)
            
            # Should return proxy object, not actual data
            assert "data_proxy" in result
            assert callable(result["data_proxy"])
            
            # Data should only be loaded when explicitly requested
            mock_img.get_fdata.assert_not_called()
            
            # Load data on demand
            actual_data = result["data_proxy"]()
            mock_img.get_fdata.assert_called_once()

    def test_chunk_loading_for_large_data(self):
        """Test chunk-based loading for memory efficiency - RED phase."""
        # This should fail because chunk loading doesn't exist yet
        loader = NiftiLoader(chunk_size=1000)  # 1000 voxels per chunk
        
        test_path = Path("/fake/path/large_file.nii.gz")
        
        with patch("nibabel.load") as mock_nib_load:
            mock_img = Mock()
            large_data = np.random.rand(64, 64, 64, 200)  # ~200MB of data
            mock_img.get_fdata.return_value = large_data
            mock_img.affine = np.eye(4)
            mock_img.header = Mock()
            mock_nib_load.return_value = mock_img
            
            result = loader.load_chunked(test_path)
            
            assert "chunk_iterator" in result
            assert hasattr(result["chunk_iterator"], "__iter__")
            assert result["total_chunks"] > 1