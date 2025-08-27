"""NIfTI file I/O operations for AccelPerm."""

from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import nibabel as nib
import numpy as np


class NiftiLoader:
    """Loader for NIfTI files with memory optimization and validation."""

    def __init__(self, lazy_loading: bool = False, chunk_size: Optional[int] = None) -> None:
        self.lazy_loading = lazy_loading
        self.chunk_size = chunk_size

    def load(self, filepath: Path, mask: Optional[Path] = None) -> Dict[str, Any]:
        """Load NIfTI file and return data with metadata."""
        try:
            img = nib.load(str(filepath))
        except FileNotFoundError:
            raise FileNotFoundError(f"NIfTI file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Invalid NIfTI file: {filepath}") from e

        if self.lazy_loading:
            result = {
                "data_proxy": lambda: img.get_fdata(),
                "affine": img.affine,
                "header": img.header,
            }
        else:
            data = img.get_fdata()
            self._check_data_integrity(data)
            
            result = {
                "data": data,
                "affine": img.affine,
                "header": img.header,
            }

            # Add shape information for 4D data
            if data.ndim == 4:
                result["n_volumes"] = data.shape[3]
                result["spatial_shape"] = data.shape[:3]

        # Handle mask if provided
        if mask is not None:
            mask_img = nib.load(str(mask))
            mask_data = mask_img.get_fdata().astype(bool)
            result["mask"] = mask_data
            result["n_voxels"] = np.sum(mask_data)
            
            if not self.lazy_loading:
                # Apply mask to data
                if data.ndim == 4:
                    masked_data = data[mask_data, :]
                else:
                    masked_data = data[mask_data]
                result["masked_data"] = masked_data

        return result

    def load_chunked(self, filepath: Path) -> Dict[str, Any]:
        """Load NIfTI file in chunks for memory efficiency."""
        try:
            img = nib.load(str(filepath))
        except FileNotFoundError:
            raise FileNotFoundError(f"NIfTI file not found: {filepath}")
        data = img.get_fdata()
        
        # Calculate number of chunks based on chunk_size
        total_voxels = np.prod(data.shape[:3])
        if self.chunk_size is None:
            chunk_size = 1000
        else:
            chunk_size = self.chunk_size
            
        total_chunks = int(np.ceil(total_voxels / chunk_size))
        
        def chunk_iterator():
            for i in range(total_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_voxels)
                # Return a chunk of data
                yield data.flat[start_idx:end_idx]
        
        return {
            "chunk_iterator": chunk_iterator(),
            "total_chunks": total_chunks,
            "affine": img.affine,
            "header": img.header,
        }

    def _check_data_integrity(self, data: np.ndarray) -> None:
        """Check data for NaN and infinite values."""
        if np.any(np.isnan(data)):
            raise ValueError("Data contains NaN values")
        if np.any(np.isinf(data)):
            raise ValueError("Data contains infinite values")

    def _validate_dimensions(self, data: np.ndarray, expected_ndim: int) -> None:
        """Validate data dimensions."""
        if data.ndim != expected_ndim:
            raise ValueError(f"Expected {expected_ndim}D data, got {data.ndim}D")

    def _validate_spatial_match(self, data1: np.ndarray, data2: np.ndarray) -> None:
        """Validate that two datasets have matching spatial dimensions."""
        if data1.shape[:3] != data2.shape[:3]:
            raise ValueError("Spatial dimensions do not match")


def load_nifti(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Load NIfTI file and return data, affine, and header."""
    img = nib.load(str(filepath))
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    return data, affine, header


def save_nifti(data: np.ndarray, affine: np.ndarray, filepath: Path) -> None:
    """Save data as NIfTI file."""
    # Create output directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Create NIfTI image and save
    img = nib.Nifti1Image(data, affine)
    img.to_filename(str(filepath))


def validate_nifti_compatibility(img1_info: Dict[str, Any], img2_info: Dict[str, Any]) -> bool:
    """Validate that two NIfTI images are compatible for processing."""
    # Check spatial dimensions match
    if img1_info["spatial_shape"] != img2_info["spatial_shape"]:
        return False
    
    # Check affine matrices are similar (allowing for small floating point differences)
    if not np.allclose(img1_info["affine"], img2_info["affine"], rtol=1e-6):
        return False
    
    return True