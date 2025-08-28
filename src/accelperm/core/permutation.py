"""Permutation engine for neuroimaging data analysis.

This module provides the core permutation functionality for statistical testing,
including sign-flipping for paired designs, full permutation enumeration,
and support for exchangeability blocks.

Based on FSL randomise implementation with GPU acceleration support.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import permutations as itertools_permutations
from pathlib import Path
from typing import Any, Literal

import numpy as np

from accelperm.utils.logging import LogLevel, setup_logger

logger = setup_logger(__name__, LogLevel.INFO)


class PermutationStrategy(ABC):
    """Abstract base class for permutation strategies."""

    def __init__(self, n_subjects: int, seed: int | None = None):
        """Initialize permutation strategy.

        Parameters
        ----------
        n_subjects : int
            Number of subjects in the dataset.
        seed : int | None
            Random seed for reproducibility.
        """
        self.n_subjects = n_subjects
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.current_permutation = 0

    @property
    @abstractmethod
    def n_possible_permutations(self) -> int:
        """Return total number of possible permutations."""
        pass

    @property
    @abstractmethod
    def permutation_type(self) -> str:
        """Return the type of permutation strategy."""
        pass

    @abstractmethod
    def next_permutation(self, permutation_number: int) -> np.ndarray:
        """Generate the next permutation.

        Parameters
        ----------
        permutation_number : int
            The permutation number to generate (1-based).

        Returns
        -------
        np.ndarray
            Permutation vector or matrix of shape (n_subjects, 1).
        """
        pass

    def apply_permutation(
        self, data: np.ndarray, permutation: np.ndarray
    ) -> np.ndarray:
        """Apply permutation to data.

        Parameters
        ----------
        data : np.ndarray
            Data matrix of shape (n_features, n_subjects).
        permutation : np.ndarray
            Permutation vector or matrix.

        Returns
        -------
        np.ndarray
            Permuted data matrix.
        """
        if self.permutation_type == "sign_flip":
            # Sign-flipping: multiply columns by �1
            return data * permutation.T
        else:
            # Full permutation: reorder columns
            perm_indices = permutation.flatten().astype(int)
            return data[:, perm_indices]


class SignFlippingStrategy(PermutationStrategy):
    """Sign-flipping permutation strategy for paired designs."""

    def __init__(
        self,
        n_subjects: int,
        seed: int | None = None,
        exhaustive: bool = False,
    ):
        """Initialize sign-flipping strategy.

        Parameters
        ----------
        n_subjects : int
            Number of subjects.
        seed : int | None
            Random seed.
        exhaustive : bool
            Whether to enumerate all possible permutations.
        """
        super().__init__(n_subjects, seed)
        self.exhaustive = exhaustive
        self._all_permutations: np.ndarray | None = None

        if exhaustive and n_subjects > 20:
            warnings.warn(
                f"Exhaustive enumeration with {n_subjects} subjects "
                f"would generate {2**n_subjects} permutations. "
                "Consider using random sampling instead.",
                stacklevel=2,
            )
            self.exhaustive = False

        if self.exhaustive:
            self._generate_all_permutations()

    @property
    def n_possible_permutations(self) -> int:
        """Return total number of possible sign-flip permutations."""
        return 2**self.n_subjects

    @property
    def permutation_type(self) -> str:
        """Return the permutation type."""
        return "sign_flip"

    def _generate_all_permutations(self) -> None:
        """Generate all possible sign-flip permutations."""
        n_perms = min(self.n_possible_permutations, 2**20)  # Limit for memory
        self._all_permutations = np.ones((self.n_subjects, n_perms))

        for i in range(1, n_perms):
            # Generate binary representation
            binary = format(i, f"0{self.n_subjects}b")
            # Convert to �1
            for j, bit in enumerate(binary):
                if bit == "1":
                    self._all_permutations[j, i] = -1

    def next_permutation(self, permutation_number: int) -> np.ndarray:
        """Generate next sign-flip permutation.

        Parameters
        ----------
        permutation_number : int
            Permutation number (1-based).

        Returns
        -------
        np.ndarray
            Sign-flip vector of shape (n_subjects, 1).
        """
        if permutation_number == 1:
            # First permutation is always unpermuted (all +1)
            return np.ones((self.n_subjects, 1))

        if (self.exhaustive and self._all_permutations is not None 
            and permutation_number <= self._all_permutations.shape[1]):
            return self._all_permutations[:, permutation_number - 1].reshape(-1, 1)

        # Random sign-flipping
        signs = self.rng.choice([-1, 1], size=self.n_subjects)
        return signs.reshape(-1, 1)


class FullPermutationStrategy(PermutationStrategy):
    """Full permutation strategy (reordering subjects)."""

    def __init__(
        self,
        n_subjects: int,
        seed: int | None = None,
        exhaustive: bool = False,
    ):
        """Initialize full permutation strategy.

        Parameters
        ----------
        n_subjects : int
            Number of subjects.
        seed : int | None
            Random seed.
        exhaustive : bool
            Whether to enumerate all permutations.
        """
        super().__init__(n_subjects, seed)
        self.exhaustive = exhaustive
        self._all_permutations: list[np.ndarray] | None = None

        if exhaustive and n_subjects > 10:
            warnings.warn(
                f"Exhaustive enumeration with {n_subjects} subjects "
                f"would generate {math.factorial(n_subjects)} permutations. "
                "Using random sampling instead.",
                stacklevel=2,
            )
            self.exhaustive = False

        if self.exhaustive:
            self._generate_all_permutations()

    @property
    def n_possible_permutations(self) -> int:
        """Return total number of possible permutations."""
        return math.factorial(self.n_subjects)

    @property
    def permutation_type(self) -> str:
        """Return the permutation type."""
        return "full_permutation"

    def _generate_all_permutations(self) -> None:
        """Generate all possible permutations."""
        base = np.arange(self.n_subjects)
        self._all_permutations = [
            np.array(perm) for perm in itertools_permutations(base)
        ]

    def next_permutation(self, permutation_number: int) -> np.ndarray:
        """Generate next permutation.

        Parameters
        ----------
        permutation_number : int
            Permutation number (1-based).

        Returns
        -------
        np.ndarray
            Permutation vector of shape (n_subjects, 1).
        """
        if permutation_number == 1:
            # First permutation is identity
            return np.arange(self.n_subjects).reshape(-1, 1)

        if (self.exhaustive and self._all_permutations is not None 
            and permutation_number <= len(self._all_permutations)):
            return self._all_permutations[permutation_number - 1].reshape(-1, 1)

        # Random permutation
        perm = self.rng.permutation(self.n_subjects)
        return perm.reshape(-1, 1)


@dataclass
class ExchangeabilityBlocks:
    """Handle exchangeability blocks for restricted permutations."""

    groups: np.ndarray
    permute_blocks: bool = False

    def __post_init__(self):
        """Validate and process exchangeability groups."""
        if len(self.groups) == 0:
            raise ValueError("Exchangeability groups cannot be empty")

        if np.any(self.groups < 0):
            raise ValueError("Group indices must be positive")

        # Convert to integer array
        self.groups = self.groups.astype(int)

        # Extract block information
        unique_groups = np.unique(self.groups)
        self.n_blocks = len(unique_groups)
        self.n_subjects = len(self.groups)

        # Store indices for each block
        self.block_indices = {}
        for group in unique_groups:
            self.block_indices[group] = np.where(self.groups == group)[0]

    @classmethod
    def from_file(cls, filepath: str | Path) -> ExchangeabilityBlocks:
        """Load exchangeability blocks from file.

        Parameters
        ----------
        filepath : str | Path
            Path to .grp file.

        Returns
        -------
        ExchangeabilityBlocks
            Loaded exchangeability blocks.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Group file not found: {filepath}")

        groups = np.loadtxt(filepath, dtype=int)
        return cls(groups)

    def generate_within_block_permutation(
        self, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Generate permutation within exchangeability blocks.

        Parameters
        ----------
        rng : np.random.Generator | None
            Random number generator.

        Returns
        -------
        np.ndarray
            Permutation vector respecting block structure.
        """
        if rng is None:
            rng = np.random.default_rng()

        perm_vector = np.arange(self.n_subjects)

        for block_idx in self.block_indices.values():
            # Permute within each block
            block_perm = rng.permutation(len(block_idx))
            perm_vector[block_idx] = block_idx[block_perm]

        return perm_vector

    def generate_block_permutation(
        self, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Generate permutation of blocks themselves.

        Parameters
        ----------
        rng : np.random.Generator | None
            Random number generator.

        Returns
        -------
        np.ndarray
            Permutation vector with blocks permuted.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Get block order permutation
        block_order = list(self.block_indices.keys())
        rng.shuffle(block_order)

        # Build permutation vector
        perm_vector = np.zeros(self.n_subjects, dtype=int)
        current_pos = 0

        for block in block_order:
            block_idx = self.block_indices[block]
            perm_vector[current_pos : current_pos + len(block_idx)] = block_idx
            current_pos += len(block_idx)

        return perm_vector


class PermutationScheduler:
    """Schedule permutation processing in batches."""

    def __init__(
        self,
        n_permutations: int,
        batch_size: int = 100,
        checkpoint_interval: int = 0,
    ):
        """Initialize permutation scheduler.

        Parameters
        ----------
        n_permutations : int
            Total number of permutations.
        batch_size : int
            Size of each batch.
        checkpoint_interval : int
            Save checkpoint every N batches (0 = no checkpointing).
        """
        self.n_permutations = n_permutations
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval

        self.n_batches = (n_permutations + batch_size - 1) // batch_size
        self.current_batch = 0
        self.completed_permutations = 0

    def has_next_batch(self) -> bool:
        """Check if there are more batches to process."""
        return self.current_batch < self.n_batches

    def next_batch(self) -> tuple[int, int]:
        """Get the next batch range.

        Returns
        -------
        tuple[int, int]
            Start and end indices for the batch.
        """
        if not self.has_next_batch():
            raise StopIteration("No more batches")

        batch_start = self.current_batch * self.batch_size
        batch_end = min(batch_start + self.batch_size, self.n_permutations)

        self.current_batch += 1
        self.completed_permutations = batch_end

        return batch_start, batch_end

    def get_progress(self) -> float:
        """Get current progress as a fraction."""
        return self.completed_permutations / self.n_permutations

    def should_checkpoint(self) -> bool:
        """Check if a checkpoint should be saved."""
        if self.checkpoint_interval == 0:
            return False
        return self.current_batch % self.checkpoint_interval == 0

    def get_checkpoint(self) -> dict[str, Any]:
        """Get current state for checkpointing.

        Returns
        -------
        dict[str, Any]
            Checkpoint data.
        """
        return {
            "current_batch": self.current_batch,
            "completed_permutations": self.completed_permutations,
            "n_permutations": self.n_permutations,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> PermutationScheduler:
        """Restore scheduler from checkpoint.

        Parameters
        ----------
        checkpoint : dict[str, Any]
            Checkpoint data.

        Returns
        -------
        PermutationScheduler
            Restored scheduler.
        """
        scheduler = cls(
            n_permutations=checkpoint["n_permutations"],
            batch_size=checkpoint["batch_size"],
        )
        scheduler.current_batch = checkpoint["current_batch"]
        scheduler.completed_permutations = checkpoint["completed_permutations"]
        return scheduler


class PermutationEngine:
    """Main permutation engine for neuroimaging analysis."""

    def __init__(
        self,
        n_subjects: int,
        n_permutations: int,
        strategy_type: Literal["sign_flip", "full"] = "sign_flip",
        exchangeability_groups: np.ndarray | None = None,
        batch_size: int = 100,
        seed: int | None = None,
    ):
        """Initialize permutation engine.

        Parameters
        ----------
        n_subjects : int
            Number of subjects.
        n_permutations : int
            Number of permutations to generate.
        strategy_type : Literal["sign_flip", "full"]
            Type of permutation strategy.
        exchangeability_groups : np.ndarray | None
            Exchangeability group assignments.
        batch_size : int
            Size of permutation batches.
        seed : int | None
            Random seed.
        """
        self.n_subjects = n_subjects
        self.n_permutations = n_permutations
        self.batch_size = batch_size
        self.seed = seed

        # Set up exchangeability blocks if provided
        self.exchangeability_blocks = None
        if exchangeability_groups is not None:
            self.exchangeability_blocks = ExchangeabilityBlocks(exchangeability_groups)

        # Create permutation strategy
        exhaustive = n_permutations < 1000 and n_subjects < 10
        if strategy_type == "sign_flip":
            self.strategy = SignFlippingStrategy(n_subjects, seed, exhaustive)
        elif strategy_type == "full":
            self.strategy = FullPermutationStrategy(n_subjects, seed, exhaustive)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        # Create scheduler
        self.scheduler = PermutationScheduler(n_permutations, batch_size)

        logger.info(
            f"Initialized {strategy_type} permutation engine: "
            f"{n_subjects} subjects, {n_permutations} permutations"
        )

    def generate_all_permutations(self) -> np.ndarray:
        """Generate all permutations at once.

        Returns
        -------
        np.ndarray
            Matrix of all permutations, shape (n_subjects, n_permutations).
        """
        permutations = np.zeros((self.n_subjects, self.n_permutations))

        for i in range(self.n_permutations):
            perm = self.strategy.next_permutation(i + 1)
            permutations[:, i] = perm.flatten()

        return permutations

    def generate_permutation_batches(self) -> Iterator[np.ndarray]:
        """Generate permutations in batches.

        Yields
        ------
        np.ndarray
            Batch of permutations, shape (n_subjects, batch_size).
        """
        while self.scheduler.has_next_batch():
            batch_start, batch_end = self.scheduler.next_batch()
            batch_size = batch_end - batch_start

            batch = np.zeros((self.n_subjects, batch_size))
            for i in range(batch_size):
                perm_num = batch_start + i + 1
                perm = self.strategy.next_permutation(perm_num)
                batch[:, i] = perm.flatten()

            yield batch

    def validate_permutations(self, permutations: np.ndarray) -> tuple[bool, str]:
        """Validate generated permutations.

        Parameters
        ----------
        permutations : np.ndarray
            Permutation matrix to validate.

        Returns
        -------
        tuple[bool, str]
            Validation result and message.
        """
        strategy_type = self.strategy.permutation_type
        return validate_permutation_matrix(permutations, strategy_type)


# Utility functions


def create_permutation_scheme(
    design_matrix: np.ndarray,
    n_permutations: int,
    exchangeability_groups: np.ndarray | None = None,
    force_sign_flip: bool = False,
) -> dict[str, Any]:
    """Create a permutation scheme based on design matrix.

    Parameters
    ----------
    design_matrix : np.ndarray
        Design matrix.
    n_permutations : int
        Number of permutations.
    exchangeability_groups : np.ndarray | None
        Exchangeability groups.
    force_sign_flip : bool
        Force sign-flipping strategy.

    Returns
    -------
    dict[str, Any]
        Permutation scheme configuration.
    """
    n_subjects = design_matrix.shape[0]

    # Determine strategy type
    # Use sign-flipping for one-sample tests or when forced
    one_sample = design_matrix.shape[1] == 1 and np.all(design_matrix == 1)
    strategy_type = "sign_flip" if (one_sample or force_sign_flip) else "full"

    return {
        "n_subjects": n_subjects,
        "n_permutations": n_permutations,
        "strategy_type": strategy_type,
        "exchangeability_groups": exchangeability_groups,
    }


def generate_sign_flip_matrix(
    n_subjects: int,
    n_permutations: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate sign-flip permutation matrix.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_permutations : int
        Number of permutations.
    seed : int | None
        Random seed.

    Returns
    -------
    np.ndarray
        Sign-flip matrix of shape (n_subjects, n_permutations).
    """
    strategy = SignFlippingStrategy(n_subjects, seed)
    matrix = np.zeros((n_subjects, n_permutations))

    for i in range(n_permutations):
        perm = strategy.next_permutation(i + 1)
        matrix[:, i] = perm.flatten()

    return matrix


def validate_permutation_matrix(
    permutations: np.ndarray,
    strategy_type: str,
) -> tuple[bool, str]:
    """Validate a permutation matrix.

    Parameters
    ----------
    permutations : np.ndarray
        Permutation matrix to validate.
    strategy_type : str
        Type of permutation strategy.

    Returns
    -------
    tuple[bool, str]
        Validation result and message.
    """
    if strategy_type == "sign_flip":
        # Check that all values are �1
        unique_values = np.unique(permutations)
        if not set(unique_values).issubset({-1, 1}):
            return False, f"Sign-flip matrix contains invalid values: {unique_values}"

        # Check first permutation is unpermuted
        if permutations.shape[1] > 0 and not np.all(permutations[:, 0] == 1):
            return False, "First permutation should be unpermuted (all +1)"

    elif strategy_type == "full" or strategy_type == "full_permutation":
        # Check each column is a valid permutation
        n_subjects = permutations.shape[0]
        for col in range(permutations.shape[1]):
            perm = permutations[:, col]
            if not set(perm).issubset(set(range(n_subjects))):
                return False, f"Invalid permutation indices in column {col}"
            if len(set(perm)) != n_subjects:
                return False, f"Duplicate indices in permutation column {col}"

    return True, "Permutation matrix is valid"
