"""Unit tests for the permutation module.

This module tests the permutation engine components including:
- Base permutation strategy interface
- Sign-flipping permutations for paired designs
- Full permutation enumeration
- Exchangeability block handling
- Permutation scheduler for batch processing
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from accelperm.core.permutation import (
    ExchangeabilityBlocks,
    FullPermutationStrategy,
    PermutationEngine,
    PermutationScheduler,
    PermutationStrategy,
    SignFlippingStrategy,
    create_permutation_scheme,
    generate_sign_flip_matrix,
    validate_permutation_matrix,
)


class TestPermutationStrategy:
    """Test the base PermutationStrategy interface."""

    def test_permutation_strategy_is_abstract(self):
        """Test that PermutationStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PermutationStrategy()

    def test_permutation_strategy_requires_implementation(self):
        """Test that concrete strategies must implement required methods."""

        class IncompleteStrategy(PermutationStrategy):
            pass

        with pytest.raises(TypeError):
            IncompleteStrategy()


class TestSignFlippingStrategy:
    """Test sign-flipping permutation strategy for paired designs."""

    def test_sign_flip_initialization(self):
        """Test SignFlippingStrategy initialization."""
        n_subjects = 10
        strategy = SignFlippingStrategy(n_subjects=n_subjects)

        assert strategy.n_subjects == n_subjects
        assert strategy.n_possible_permutations == 2**n_subjects
        assert strategy.permutation_type == "sign_flip"

    def test_sign_flip_generates_correct_matrix(self):
        """Test that sign-flipping generates correct permutation matrix."""
        n_subjects = 5
        strategy = SignFlippingStrategy(n_subjects=n_subjects)

        # Generate first permutation (should be all ones)
        perm_matrix = strategy.next_permutation(permutation_number=1)
        assert perm_matrix.shape == (n_subjects, 1)
        assert np.all(perm_matrix == 1)

        # Generate random permutation
        perm_matrix = strategy.next_permutation(permutation_number=10)
        assert perm_matrix.shape == (n_subjects, 1)
        assert np.all(np.abs(perm_matrix) == 1)  # All values should be ±1
        assert set(np.unique(perm_matrix)) <= {-1, 1}

    def test_sign_flip_with_seed_reproducibility(self):
        """Test that sign-flipping with seed is reproducible."""
        n_subjects = 10
        seed = 42

        strategy1 = SignFlippingStrategy(n_subjects=n_subjects, seed=seed)
        perm1 = strategy1.next_permutation(permutation_number=5)

        strategy2 = SignFlippingStrategy(n_subjects=n_subjects, seed=seed)
        perm2 = strategy2.next_permutation(permutation_number=5)

        assert np.array_equal(perm1, perm2)

    def test_sign_flip_exhaustive_enumeration(self):
        """Test exhaustive enumeration for small number of subjects."""
        n_subjects = 3
        strategy = SignFlippingStrategy(n_subjects=n_subjects, exhaustive=True)

        # Should generate exactly 2^3 = 8 unique permutations
        permutations = []
        for i in range(8):
            perm = strategy.next_permutation(permutation_number=i + 1)
            permutations.append(perm.flatten())

        # Check all permutations are unique
        unique_perms = np.unique(np.array(permutations), axis=0)
        assert len(unique_perms) == 8

    def test_sign_flip_apply_to_data(self):
        """Test applying sign-flip permutation to data."""
        n_subjects = 5
        n_voxels = 100
        data = np.random.randn(n_voxels, n_subjects)

        strategy = SignFlippingStrategy(n_subjects=n_subjects)
        perm_matrix = strategy.next_permutation(permutation_number=2)

        # Apply permutation
        permuted_data = strategy.apply_permutation(data, perm_matrix)

        assert permuted_data.shape == data.shape
        # Check that signs have been flipped correctly
        for i in range(n_subjects):
            if perm_matrix[i] == -1:
                assert np.array_equal(permuted_data[:, i], -data[:, i])
            else:
                assert np.array_equal(permuted_data[:, i], data[:, i])


class TestFullPermutationStrategy:
    """Test full permutation strategy."""

    def test_full_permutation_initialization(self):
        """Test FullPermutationStrategy initialization."""
        n_subjects = 5
        strategy = FullPermutationStrategy(n_subjects=n_subjects)

        assert strategy.n_subjects == n_subjects
        assert strategy.n_possible_permutations == math.factorial(n_subjects)
        assert strategy.permutation_type == "full_permutation"

    def test_full_permutation_generates_valid_permutations(self):
        """Test that full permutation generates valid permutation vectors."""
        n_subjects = 4
        strategy = FullPermutationStrategy(n_subjects=n_subjects)

        # First permutation should be identity
        perm_vector = strategy.next_permutation(permutation_number=1)
        assert perm_vector.shape == (n_subjects, 1)
        assert np.array_equal(perm_vector.flatten(), np.arange(n_subjects))

        # Random permutation should be a valid permutation
        perm_vector = strategy.next_permutation(permutation_number=5)
        assert perm_vector.shape == (n_subjects, 1)
        assert set(perm_vector.flatten()) == set(range(n_subjects))

    def test_full_permutation_exhaustive(self):
        """Test exhaustive enumeration of all permutations."""
        n_subjects = 3
        strategy = FullPermutationStrategy(n_subjects=n_subjects, exhaustive=True)

        # Should generate exactly 3! = 6 unique permutations
        permutations = []
        for i in range(6):
            perm = strategy.next_permutation(permutation_number=i + 1)
            permutations.append(perm.flatten())

        # Check all permutations are unique
        unique_perms = [tuple(p) for p in permutations]
        assert len(set(unique_perms)) == 6

    def test_full_permutation_apply_to_data(self):
        """Test applying full permutation to data."""
        n_subjects = 4
        n_voxels = 50
        data = np.random.randn(n_voxels, n_subjects)

        strategy = FullPermutationStrategy(n_subjects=n_subjects)
        perm_vector = np.array([[2], [0], [3], [1]])  # Specific permutation

        # Apply permutation
        permuted_data = strategy.apply_permutation(data, perm_vector)

        assert permuted_data.shape == data.shape
        # Check that columns have been permuted correctly
        assert np.array_equal(permuted_data[:, 0], data[:, 2])
        assert np.array_equal(permuted_data[:, 1], data[:, 0])
        assert np.array_equal(permuted_data[:, 2], data[:, 3])
        assert np.array_equal(permuted_data[:, 3], data[:, 1])


class TestExchangeabilityBlocks:
    """Test exchangeability block handling."""

    def test_exchangeability_blocks_initialization(self):
        """Test ExchangeabilityBlocks initialization."""
        # Groups: [1, 1, 2, 2, 3, 3]
        groups = np.array([1, 1, 2, 2, 3, 3])
        blocks = ExchangeabilityBlocks(groups)

        assert blocks.n_subjects == 6
        assert blocks.n_blocks == 3
        assert len(blocks.block_indices) == 3

    def test_exchangeability_blocks_from_file(self):
        """Test loading exchangeability blocks from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".grp", delete=False) as f:
            f.write("1\n1\n2\n2\n3\n3\n")
            grp_file = f.name

        try:
            blocks = ExchangeabilityBlocks.from_file(grp_file)
            assert blocks.n_subjects == 6
            assert blocks.n_blocks == 3
        finally:
            Path(grp_file).unlink()

    def test_exchangeability_blocks_validation(self):
        """Test validation of exchangeability blocks."""
        # Invalid: empty blocks
        with pytest.raises(ValueError, match="empty"):
            ExchangeabilityBlocks(np.array([]))

        # Invalid: negative group numbers
        with pytest.raises(ValueError, match="positive"):
            ExchangeabilityBlocks(np.array([1, 2, -1, 3]))

    def test_exchangeability_blocks_within_block_permutation(self):
        """Test permutation within exchangeability blocks."""
        groups = np.array([1, 1, 1, 2, 2, 2])
        blocks = ExchangeabilityBlocks(groups)

        perm_vector = blocks.generate_within_block_permutation()

        # Check that subjects are only permuted within their blocks
        assert set(perm_vector[:3]) == {0, 1, 2}  # Block 1
        assert set(perm_vector[3:]) == {3, 4, 5}  # Block 2

    def test_exchangeability_blocks_between_block_permutation(self):
        """Test permutation of blocks themselves."""
        groups = np.array([1, 1, 2, 2, 3, 3])
        blocks = ExchangeabilityBlocks(groups, permute_blocks=True)

        # This should permute entire blocks
        perm_vector = blocks.generate_block_permutation()

        # Verify blocks are preserved but may be reordered
        assert perm_vector.shape == (6,)
        # Check that paired subjects stay together
        for i in range(0, 6, 2):
            assert abs(perm_vector[i] - perm_vector[i + 1]) == 1


class TestPermutationScheduler:
    """Test permutation scheduler for batch processing."""

    def test_scheduler_initialization(self):
        """Test PermutationScheduler initialization."""
        n_permutations = 10000
        batch_size = 100

        scheduler = PermutationScheduler(
            n_permutations=n_permutations, batch_size=batch_size
        )

        assert scheduler.n_permutations == n_permutations
        assert scheduler.batch_size == batch_size
        assert scheduler.n_batches == 100
        assert scheduler.current_batch == 0

    def test_scheduler_batch_generation(self):
        """Test batch generation by scheduler."""
        scheduler = PermutationScheduler(n_permutations=1000, batch_size=100)

        batches_generated = 0
        while scheduler.has_next_batch():
            batch_start, batch_end = scheduler.next_batch()
            assert batch_end - batch_start <= 100
            batches_generated += 1

        assert batches_generated == 10

    def test_scheduler_handles_uneven_batches(self):
        """Test scheduler with uneven batch sizes."""
        scheduler = PermutationScheduler(n_permutations=1050, batch_size=100)

        batches = []
        while scheduler.has_next_batch():
            batches.append(scheduler.next_batch())

        assert len(batches) == 11
        assert batches[-1] == (1000, 1050)  # Last batch has 50 permutations

    def test_scheduler_progress_tracking(self):
        """Test progress tracking in scheduler."""
        scheduler = PermutationScheduler(n_permutations=500, batch_size=100)

        assert scheduler.get_progress() == 0.0

        scheduler.next_batch()
        assert scheduler.get_progress() == 0.2

        scheduler.next_batch()
        assert scheduler.get_progress() == 0.4

    def test_scheduler_checkpointing(self):
        """Test checkpointing functionality."""
        scheduler = PermutationScheduler(
            n_permutations=1000, batch_size=100, checkpoint_interval=3
        )

        # Process 3 batches
        for _ in range(3):
            scheduler.next_batch()

        assert scheduler.should_checkpoint()
        checkpoint = scheduler.get_checkpoint()
        assert checkpoint["current_batch"] == 3
        assert checkpoint["completed_permutations"] == 300

    def test_scheduler_restore_from_checkpoint(self):
        """Test restoring scheduler from checkpoint."""
        checkpoint = {
            "current_batch": 5,
            "completed_permutations": 500,
            "n_permutations": 1000,
            "batch_size": 100,
        }

        scheduler = PermutationScheduler.from_checkpoint(checkpoint)

        assert scheduler.current_batch == 5
        assert scheduler.get_progress() == 0.5


class TestPermutationEngine:
    """Test the main permutation engine."""

    def test_engine_initialization(self):
        """Test PermutationEngine initialization."""
        n_subjects = 20
        n_permutations = 1000

        engine = PermutationEngine(
            n_subjects=n_subjects,
            n_permutations=n_permutations,
            strategy_type="sign_flip",
        )

        assert engine.n_subjects == n_subjects
        assert engine.n_permutations == n_permutations
        assert isinstance(engine.strategy, SignFlippingStrategy)

    def test_engine_with_exchangeability_blocks(self):
        """Test engine with exchangeability blocks."""
        groups = np.array([1, 1, 2, 2, 3, 3])

        engine = PermutationEngine(
            n_subjects=6,
            n_permutations=100,
            strategy_type="full",
            exchangeability_groups=groups,
        )

        assert engine.exchangeability_blocks is not None
        assert engine.exchangeability_blocks.n_blocks == 3

    def test_engine_generate_all_permutations(self):
        """Test generating all permutations."""
        engine = PermutationEngine(
            n_subjects=10, n_permutations=100, strategy_type="sign_flip", batch_size=25
        )

        all_permutations = engine.generate_all_permutations()

        assert all_permutations.shape == (10, 100)
        assert np.all(np.abs(all_permutations) == 1)  # All values should be ±1

    def test_engine_batch_processing(self):
        """Test batch processing of permutations."""
        engine = PermutationEngine(
            n_subjects=5, n_permutations=50, strategy_type="sign_flip", batch_size=10
        )

        batches_processed = 0
        for batch in engine.generate_permutation_batches():
            assert batch.shape[0] == 5  # n_subjects
            assert batch.shape[1] <= 10  # batch_size
            batches_processed += 1

        assert batches_processed == 5

    def test_engine_validation(self):
        """Test validation of permutation matrices."""
        engine = PermutationEngine(
            n_subjects=10, n_permutations=100, strategy_type="sign_flip"
        )

        permutations = engine.generate_all_permutations()

        # Validate permutations
        is_valid, message = engine.validate_permutations(permutations)
        assert is_valid
        assert "valid" in message.lower()

    def test_engine_statistical_properties(self):
        """Test statistical properties of generated permutations."""
        n_subjects = 10
        n_permutations = 10000

        engine = PermutationEngine(
            n_subjects=n_subjects,
            n_permutations=n_permutations,
            strategy_type="sign_flip",
            seed=42,
        )

        permutations = engine.generate_all_permutations()

        # Check that mean is close to 0 for sign-flipping
        mean_flips = np.mean(permutations, axis=1)
        assert np.all(np.abs(mean_flips) < 0.1)  # Should be close to 0

        # Check that first permutation is unpermuted (all ones)
        assert np.all(permutations[:, 0] == 1)


class TestPermutationUtilityFunctions:
    """Test utility functions for permutation module."""

    def test_create_permutation_scheme(self):
        """Test creating a permutation scheme."""
        design_matrix = np.ones((10, 3))

        scheme = create_permutation_scheme(
            design_matrix=design_matrix,
            n_permutations=100,
            force_sign_flip=False,
        )

        assert scheme["n_subjects"] == 10
        assert scheme["n_permutations"] == 100
        assert "strategy_type" in scheme

    def test_generate_sign_flip_matrix(self):
        """Test generating sign-flip matrix."""
        n_subjects = 5
        n_permutations = 10

        flip_matrix = generate_sign_flip_matrix(
            n_subjects=n_subjects, n_permutations=n_permutations, seed=42
        )

        assert flip_matrix.shape == (n_subjects, n_permutations)
        assert np.all(np.abs(flip_matrix) == 1)
        assert np.all(flip_matrix[:, 0] == 1)  # First is unpermuted

    def test_validate_permutation_matrix(self):
        """Test validation of permutation matrices."""
        # Valid sign-flip matrix (first column all 1s, others are valid ±1)
        valid_sign_flip = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, 1]]).T
        is_valid, msg = validate_permutation_matrix(valid_sign_flip, "sign_flip")
        assert is_valid

        # Invalid sign-flip matrix (contains 0)
        invalid_sign_flip = np.array([[1, 1, 1], [0, 1, -1], [1, -1, 1]]).T
        is_valid, msg = validate_permutation_matrix(invalid_sign_flip, "sign_flip")
        assert not is_valid
        assert "sign-flip" in msg.lower()

        # Valid full permutation matrix
        valid_full = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]).T
        is_valid, msg = validate_permutation_matrix(valid_full, "full")
        assert is_valid

        # Invalid full permutation (duplicate indices)
        invalid_full = np.array([[0, 1, 2], [1, 1, 0], [2, 0, 1]]).T
        is_valid, msg = validate_permutation_matrix(invalid_full, "full")
        assert not is_valid
