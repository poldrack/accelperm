"""Output writer operations for AccelPerm."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from accelperm.io.nifti import save_nifti


class OutputWriter:
    """Writer for neuroimaging analysis results with format compatibility."""

    def __init__(
        self,
        format_style: str = "standard",
        include_metadata: bool = False,
        validate_output: bool = False,
    ) -> None:
        self.format_style = format_style
        self.include_metadata = include_metadata
        self.validate_output = validate_output

    def save_statistical_map(
        self,
        stat_map: np.ndarray,
        affine: np.ndarray,
        output_path: Path,
        map_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save statistical map to NIfTI format."""
        if self.validate_output:
            self._validate_statistical_map(stat_map)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the NIfTI file
        save_nifti(stat_map, affine, output_path)

    def save_p_value_map(
        self, p_values: np.ndarray, affine: np.ndarray, output_path: Path
    ) -> None:
        """Save p-value map to NIfTI format."""
        if self.validate_output:
            self._validate_p_value_map(p_values)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the NIfTI file
        save_nifti(p_values, affine, output_path)

    def save_corrected_p_values(
        self,
        corrected_p: np.ndarray,
        affine: np.ndarray,
        output_path: Path,
        correction_method: str,
    ) -> None:
        """Save corrected p-value map to NIfTI format."""
        if self.validate_output:
            self._validate_p_value_map(corrected_p)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the NIfTI file
        save_nifti(corrected_p, affine, output_path)

    def save_tfce_map(
        self, tfce_map: np.ndarray, affine: np.ndarray, output_path: Path
    ) -> None:
        """Save TFCE-enhanced statistical map to NIfTI format."""
        if self.validate_output:
            self._validate_statistical_map(tfce_map)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the NIfTI file
        save_nifti(tfce_map, affine, output_path)

    def generate_cluster_table(
        self, clusters: list[dict[str, Any]], output_path: Path
    ) -> None:
        """Generate cluster summary table in text format."""
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            # Write header
            f.write("# Cluster Table\n")
            f.write("# Size\tPeak_X\tPeak_Y\tPeak_Z\tPeak_Stat\tP_Corrected\n")

            # Write cluster data
            for cluster in clusters:
                size = cluster["size"]
                peak_coord = cluster["peak_coord"]
                peak_stat = cluster["peak_stat"]
                p_corrected = cluster["p_corrected"]

                f.write(
                    f"{size}\t{peak_coord[0]}\t{peak_coord[1]}\t{peak_coord[2]}\t"
                    f"{peak_stat:.3f}\t{p_corrected:.6f}\n"
                )

    def save_multiple_maps(
        self,
        maps: dict[str, np.ndarray],
        affine: np.ndarray,
        output_dir: Path,
    ) -> None:
        """Save multiple statistical maps in batch."""
        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)

        for map_name, map_data in maps.items():
            output_path = output_dir / f"{map_name}.nii.gz"
            if self.validate_output:
                self._validate_statistical_map(map_data)
            save_nifti(map_data, affine, output_path)

    def check_output_completeness(
        self, output_dir: Path, expected_files: list[str]
    ) -> bool:
        """Check if all expected output files exist."""
        for filename in expected_files:
            file_path = output_dir / filename
            if not file_path.exists():
                return False
        return True

    def create_analysis_log(
        self, log_data: dict[str, Any], output_path: Path
    ) -> None:
        """Create comprehensive analysis log in JSON format."""
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(log_data, f, indent=2, default=str)

    def _validate_statistical_map(self, stat_map: np.ndarray) -> None:
        """Validate statistical map for invalid values."""
        if np.any(np.isnan(stat_map)):
            raise ValueError("Statistical map contains invalid values (NaN)")
        if np.any(np.isinf(stat_map)):
            raise ValueError("Statistical map contains invalid values (Inf)")

    def _validate_p_value_map(self, p_values: np.ndarray) -> None:
        """Validate p-value map for proper range."""
        if np.any(p_values < 0) or np.any(p_values > 1):
            raise ValueError("P-values must be between 0 and 1")
        if np.any(np.isnan(p_values)):
            raise ValueError("P-values contain invalid values (NaN)")


def save_statistical_map(
    stat_map: np.ndarray, affine: np.ndarray, output_path: Path
) -> None:
    """Save statistical map to NIfTI format."""
    writer = OutputWriter()
    writer.save_statistical_map(stat_map, affine, output_path, "statistical_map")


def save_p_value_map(
    p_values: np.ndarray, affine: np.ndarray, output_path: Path
) -> None:
    """Save p-value map to NIfTI format."""
    writer = OutputWriter()
    writer.save_p_value_map(p_values, affine, output_path)


def generate_cluster_table(
    clusters: list[dict[str, Any]], output_path: Path
) -> None:
    """Generate cluster summary table in text format."""
    writer = OutputWriter()
    writer.generate_cluster_table(clusters, output_path)


def create_results_summary(
    results: dict[str, Any], output_path: Path
) -> None:
    """Create results summary in text format."""
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Analysis Results Summary\n")
        f.write("# Generated by AccelPerm\n\n")

        for key, value in results.items():
            f.write(f"{key}: {value}\n")