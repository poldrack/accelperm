"""Contrast file I/O operations for AccelPerm."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ContrastLoader:
    """Loader for contrast matrices with validation and format support."""

    def __init__(
        self,
        format_style: str = "standard",
        validate_contrasts: bool = False,
    ) -> None:
        self.format_style = format_style
        self.validate_contrasts = validate_contrasts

    def load(
        self,
        filepath: Path,
        contrast_names: list[str] | None = None,
        design_info: dict[str, Any] | None = None,
        check_rank: bool = False,
    ) -> dict[str, Any]:
        """Load contrast matrix from file and return data with metadata."""
        try:
            # Load data based on file format
            if filepath.suffix.lower() == ".con":
                # FSL format: space-separated, no headers
                data = pd.read_csv(filepath, sep=" ", header=None)
            elif filepath.suffix.lower() == ".csv":
                # CSV format with headers
                data = pd.read_csv(filepath)
                # If first column is 'name', use it for contrast names
                if "name" in data.columns:
                    if contrast_names is None:
                        contrast_names = data["name"].tolist()
                    data = data.drop("name", axis=1)
            elif filepath.suffix.lower() in [".txt", ".mat"]:
                # Tab or space-separated format
                # Handle inconsistent row lengths manually
                with open(filepath) as f:
                    lines = f.readlines()

                # Parse lines manually to handle inconsistent lengths
                rows = []
                max_cols = 0

                for line in lines:
                    line = line.strip()
                    if line:  # Skip empty lines
                        if "\t" in line:
                            values = line.split("\t")
                        else:
                            values = line.split()

                        # Check for inconsistent number of columns
                        if (
                            self.validate_contrasts
                            and max_cols > 0
                            and len(values) != max_cols
                        ):
                            raise ValueError(
                                "Inconsistent number of regressors in contrast file"
                            )

                        max_cols = max(max_cols, len(values))
                        rows.append([float(v) for v in values])

                # Convert to DataFrame
                data = pd.DataFrame(rows)
            else:
                raise ValueError(f"Unsupported contrast file format: {filepath.suffix}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Contrast file not found: {filepath}") from None
        except ValueError:
            raise  # Re-raise ValueError with original message
        except Exception as e:
            raise ValueError(f"Error loading contrast file: {filepath}") from e

        # Convert to numpy array
        contrast_matrix = data.values.astype(float)

        # Note: dimension validation is handled during parsing for text files

        # Generate default contrast names if not provided
        if contrast_names is None:
            if self.format_style == "fsl":
                contrast_names = [f"C{i+1}" for i in range(contrast_matrix.shape[0])]
            else:
                contrast_names = [
                    f"contrast_{i+1}" for i in range(contrast_matrix.shape[0])
                ]

        # Validate contrasts
        if self.validate_contrasts:
            self._validate_contrast_matrix(contrast_matrix)

        # Check rank if requested
        if check_rank:
            rank = np.linalg.matrix_rank(contrast_matrix)
            if rank < contrast_matrix.shape[0]:
                raise ValueError("Contrast matrix is rank deficient")

        # Check design compatibility if provided
        design_compatible = True
        if design_info is not None:
            design_compatible, _ = validate_contrast_compatibility(
                contrast_matrix, design_info
            )

        # Prepare result
        result = {
            "contrast_matrix": contrast_matrix,
            "contrast_names": contrast_names,
            "n_contrasts": contrast_matrix.shape[0],
            "n_regressors": contrast_matrix.shape[1],
        }

        if self.format_style != "standard":
            result["format_style"] = self.format_style

        if design_info is not None:
            result["design_compatible"] = design_compatible

        return result

    def load_multiple(self, filepaths: list[Path]) -> dict[str, Any]:
        """Load multiple contrast files and combine them."""
        all_contrasts = []
        all_names = []

        for filepath in filepaths:
            result = self.load(filepath)
            all_contrasts.append(result["contrast_matrix"])
            all_names.extend(result["contrast_names"])

        # Combine all contrast matrices
        combined_matrix = np.vstack(all_contrasts)

        return {
            "contrast_matrix": combined_matrix,
            "contrast_names": all_names,
            "n_contrasts": combined_matrix.shape[0],
            "n_regressors": combined_matrix.shape[1],
        }

    def create_standard_contrasts(self, column_names: list[str]) -> dict[str, Any]:
        """Create standard contrast sets for common analyses."""
        contrasts = {"main_effects": [], "interactions": []}

        # Create main effect contrasts (excluding intercept)
        for i, col_name in enumerate(column_names):
            if col_name.lower() != "intercept":
                contrast = np.zeros(len(column_names))
                contrast[i] = 1
                contrasts["main_effects"].append(
                    {"name": f"{col_name}_effect", "contrast": contrast}
                )

        # Create interaction contrasts (for columns containing 'x' or '_x_')
        for i, col_name in enumerate(column_names):
            if "x" in col_name.lower() or "_x_" in col_name.lower():
                contrast = np.zeros(len(column_names))
                contrast[i] = 1
                contrasts["interactions"].append(
                    {"name": f"{col_name}_interaction", "contrast": contrast}
                )

        return contrasts

    def create_polynomial_contrasts(self, n_levels: int) -> dict[str, np.ndarray]:
        """Create polynomial contrasts for ordered factors."""
        contrasts = {}

        if n_levels >= 2:
            # Linear contrast
            linear = np.linspace(-1, 1, n_levels)
            contrasts["linear"] = linear

        if n_levels >= 3:
            # Quadratic contrast
            x = np.linspace(-1, 1, n_levels)
            quadratic = x**2 - np.mean(x**2)
            contrasts["quadratic"] = quadratic

        if n_levels >= 4:
            # Cubic contrast
            cubic = x**3 - np.mean(x**3)
            contrasts["cubic"] = cubic

        return contrasts

    def _validate_contrast_matrix(self, contrast_matrix: np.ndarray) -> None:
        """Validate contrast matrix for common issues."""
        # Check for all-zero rows
        zero_rows = np.all(contrast_matrix == 0, axis=1)
        if np.any(zero_rows):
            raise ValueError("Contrast contains all-zero rows")

        # Check for NaN or infinite values
        if np.any(np.isnan(contrast_matrix)):
            raise ValueError("Contrast matrix contains NaN values")
        if np.any(np.isinf(contrast_matrix)):
            raise ValueError("Contrast matrix contains infinite values")


def load_contrast_matrix(filepath: Path) -> tuple[np.ndarray, list[str]]:
    """Load contrast matrix from file and return matrix and names."""
    loader = ContrastLoader()
    result = loader.load(filepath)
    return result["contrast_matrix"], result["contrast_names"]


def validate_contrast_compatibility(
    contrast_matrix: np.ndarray, design_info: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Validate compatibility between contrast matrix and design matrix."""
    issues = []

    # Check number of regressors matches
    if contrast_matrix.shape[1] != design_info["n_regressors"]:
        issues.append(
            f"Number of regressors mismatch: contrast has {contrast_matrix.shape[1]}, "
            f"design has {design_info['n_regressors']}"
        )

    # Check for proper orthogonality (if design matrix is orthogonal)
    if "orthogonal" in design_info and design_info["orthogonal"]:
        # For orthogonal designs, contrasts should sum to zero for proper interpretation
        for i, contrast_row in enumerate(contrast_matrix):
            if not np.isclose(np.sum(contrast_row), 0, atol=1e-10):
                issues.append(
                    f"Contrast {i+1} does not sum to zero (may not be interpretable)"
                )

    is_compatible = len(issues) == 0
    return is_compatible, issues


def create_t_contrast(
    column_names: list[str], contrast_spec: dict[str, float]
) -> np.ndarray:
    """Create t-contrast vector from column names and specification."""
    n_regressors = len(column_names)
    contrast = np.zeros(n_regressors)

    for col_name, weight in contrast_spec.items():
        if col_name in column_names:
            idx = column_names.index(col_name)
            contrast[idx] = weight
        else:
            raise ValueError(f"Column '{col_name}' not found in design matrix")

    return contrast


def create_f_contrast(
    column_names: list[str], contrast_specs: list[dict[str, float]]
) -> np.ndarray:
    """Create F-contrast matrix from column names and list of contrast specifications."""
    n_regressors = len(column_names)
    n_contrasts = len(contrast_specs)
    f_contrast = np.zeros((n_contrasts, n_regressors))

    for i, contrast_spec in enumerate(contrast_specs):
        f_contrast[i, :] = create_t_contrast(column_names, contrast_spec)

    return f_contrast
