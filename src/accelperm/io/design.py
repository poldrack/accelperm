"""Design matrix I/O operations for AccelPerm."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class DesignMatrixLoader:
    """Loader for design matrices with validation and preprocessing."""

    def __init__(
        self,
        encode_categorical: bool = False,
        add_intercept: bool = False,
        format_style: str = "standard",
        standardize: bool = False,
        orthogonalize: bool = False,
    ) -> None:
        self.encode_categorical = encode_categorical
        self.add_intercept = add_intercept
        self.format_style = format_style
        self.standardize = standardize
        self.orthogonalize = orthogonalize

    def load(
        self,
        filepath: Path,
        check_rank: bool = False,
        warn_constant: bool = False,
    ) -> dict[str, Any]:
        """Load design matrix from file and return data with metadata."""
        try:
            # Load data based on file format
            if filepath.suffix.lower() == ".csv":
                data = pd.read_csv(filepath)
            elif filepath.suffix.lower() == ".tsv":
                data = pd.read_csv(filepath, sep="\t")
            elif filepath.suffix.lower() == ".mat" and self.format_style == "fsl":
                # FSL format: space-separated, no headers
                data = pd.read_csv(filepath, sep=" ", header=None)
                data.columns = [f"EV{i+1}" for i in range(len(data.columns))]
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Design matrix file not found: {filepath}"
            ) from None
        except ValueError:
            raise  # Re-raise ValueError with original message
        except Exception as e:
            raise ValueError(f"Error loading design matrix: {filepath}") from e

        # Check for missing values
        if data.isnull().any().any():
            raise ValueError("Design matrix contains missing values")

        # Store original column names
        original_columns = list(data.columns)

        # Handle categorical encoding
        categorical_columns = []

        # Identify categorical columns
        for col in data.columns:
            if data[col].dtype == "object" or data[col].dtype.name == "category":
                categorical_columns.append(col)

        # Process categorical columns
        if categorical_columns:
            if self.encode_categorical:
                # Apply one-hot encoding
                for col in categorical_columns:
                    dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                    data = pd.concat([data.drop(col, axis=1), dummies], axis=1)
            else:
                # Drop categorical columns if not encoding
                for col in categorical_columns:
                    data = data.drop(col, axis=1)

        # Convert to numpy array
        design_matrix = data.values.astype(float)

        # Add intercept column if requested
        column_names = list(data.columns)
        if self.add_intercept:
            intercept = np.ones((design_matrix.shape[0], 1))
            design_matrix = np.column_stack([intercept, design_matrix])
            column_names = ["intercept"] + column_names

        # Check for warnings
        warnings_list = []
        if warn_constant:
            for i, col_name in enumerate(column_names):
                col_data = design_matrix[:, i]
                if np.std(col_data) < 1e-10:
                    warnings_list.append(f"Column '{col_name}' is constant")

        # Standardize continuous regressors
        if self.standardize:
            for i, col_name in enumerate(column_names):
                if col_name != "intercept" and col_name not in categorical_columns:
                    col_data = design_matrix[:, i]
                    if np.std(col_data) > 1e-10:  # Avoid division by zero
                        design_matrix[:, i] = (col_data - np.mean(col_data)) / np.std(
                            col_data
                        )

        # Orthogonalize regressors using Gram-Schmidt process
        if self.orthogonalize:
            design_matrix = self._gram_schmidt(design_matrix)

        # Check rank deficiency
        if check_rank:
            rank = np.linalg.matrix_rank(design_matrix)
            if rank < design_matrix.shape[1]:
                raise ValueError("Design matrix is rank deficient")

        # Prepare result
        result = {
            "design_matrix": design_matrix,
            "column_names": column_names,
            "n_subjects": design_matrix.shape[0],
            "n_regressors": design_matrix.shape[1],
        }

        if categorical_columns:
            result["categorical_columns"] = categorical_columns

        if warnings_list:
            result["warnings"] = warnings_list

        if self.format_style != "standard":
            result["format_style"] = self.format_style

        return result

    def _gram_schmidt(self, matrix: np.ndarray) -> np.ndarray:
        """Orthogonalize matrix columns using Gram-Schmidt process."""
        orthogonal_matrix = matrix.copy()
        n_cols = matrix.shape[1]

        for i in range(n_cols):
            for j in range(i):
                # Project vector i onto vector j and subtract
                projection = (
                    np.dot(orthogonal_matrix[:, i], orthogonal_matrix[:, j])
                    / np.dot(orthogonal_matrix[:, j], orthogonal_matrix[:, j])
                    * orthogonal_matrix[:, j]
                )
                orthogonal_matrix[:, i] -= projection

            # Normalize (optional, for orthonormal basis)
            norm = np.linalg.norm(orthogonal_matrix[:, i])
            if norm > 1e-10:
                orthogonal_matrix[:, i] /= norm

        return orthogonal_matrix


def load_design_matrix(filepath: Path) -> tuple[np.ndarray, list[str]]:
    """Load design matrix from file and return matrix and column names."""
    loader = DesignMatrixLoader()
    result = loader.load(filepath)
    return result["design_matrix"], result["column_names"]


def validate_design_matrix(design_matrix: np.ndarray) -> tuple[bool, list[str]]:
    """Validate design matrix and return validation status and issues."""
    issues = []

    # Check for NaN values
    if np.any(np.isnan(design_matrix)):
        issues.append("Design matrix contains NaN values")

    # Check for infinite values
    if np.any(np.isinf(design_matrix)):
        issues.append("Design matrix contains infinite values")

    # Check for rank deficiency
    try:
        rank = np.linalg.matrix_rank(design_matrix)
        if rank < design_matrix.shape[1]:
            issues.append(f"Design matrix is rank deficient (rank={rank})")
    except np.linalg.LinAlgError:
        issues.append("Could not compute matrix rank")

    # Check for constant columns (excluding likely intercept in first column)
    for i in range(1, design_matrix.shape[1]):  # Skip first column (likely intercept)
        col = design_matrix[:, i]
        if np.std(col) < 1e-10:
            issues.append(f"Column {i} is constant")

    is_valid = len(issues) == 0
    return is_valid, issues


def create_contrast_matrix(
    column_names: list[str], contrast_dict: dict[str, float]
) -> np.ndarray:
    """Create contrast matrix from column names and contrast specification."""
    n_cols = len(column_names)
    contrast = np.zeros(n_cols)

    for col_name, weight in contrast_dict.items():
        if col_name in column_names:
            idx = column_names.index(col_name)
            contrast[idx] = weight
        else:
            raise ValueError(f"Column '{col_name}' not found in design matrix")

    return contrast
