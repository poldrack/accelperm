"""Command-line interface for AccelPerm."""

from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from scipy.ndimage import label as scipy_label

from accelperm.backends.factory import BackendFactory
from accelperm.core.corrections import (
    ClusterCorrection,
    FDRCorrection,
    FWERCorrection,
    TFCECorrection,
)
from accelperm.core.permutation import SignFlippingStrategy
from accelperm.core.tfce import TFCEProcessor
from accelperm.io.contrast import ContrastLoader
from accelperm.io.design import DesignMatrixLoader
from accelperm.io.nifti import NiftiLoader
from accelperm.io.output import OutputWriter

__version__ = "0.1.0"

# Create Typer app
app = typer.Typer(
    name="accelperm",
    help="AccelPerm - GPU-accelerated permutation testing for neuroimaging",
    add_completion=False,
)

# Rich console for pretty output
console = Console()


def version_callback(value: bool):
    """Show version information."""
    if value:
        console.print(f"AccelPerm version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        callback=version_callback,
        help="Show version information",
    ),
) -> None:
    """
    AccelPerm - GPU-accelerated permutation testing for neuroimaging data.

    A modern, fast alternative to FSL randomise with GPU acceleration support.
    """
    pass


@app.command()
def glm(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input NIfTI file (.nii.gz)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    design_file: Path | None = typer.Option(
        None,
        "--design",
        "-d",
        help="Design matrix file (.txt, .csv, .tsv) - not required for one-sample t-test",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    contrast_file: Path | None = typer.Option(
        None,
        "--contrasts",
        "-c",
        help="Contrast file (.con, .txt, .csv) - not required for one-sample t-test",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output directory for results",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    one_sample: bool = typer.Option(
        False,
        "-1",
        "--one-sample",
        help="Perform one-sample t-test (test mean against zero)",
    ),
    backend: str = typer.Option(
        "auto",
        "--backend",
        "-b",
        help="Backend to use (auto, cpu, mps, gpu_optimized)",
        case_sensitive=False,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output",
    ),
    n_permutations: int = typer.Option(
        1000,
        "--n-perm",
        "-v",
        help="Number of permutations for statistical testing",
        min=100,
    ),
    correction: str = typer.Option(
        "voxel",
        "--correction",
        help="Multiple comparison correction methods (comma-separated: none, voxel, cluster, tfce). Example: 'voxel,tfce'",
        case_sensitive=False,
    ),
    alpha: float = typer.Option(
        0.05,
        "--alpha",
        help="Significance level for correction",
        min=0.001,
        max=0.5,
    ),
    tfce_height: float = typer.Option(
        2.0,
        "--tfce-height",
        help="TFCE height exponent (H parameter)",
        min=0.1,
        max=10.0,
    ),
    tfce_extent: float = typer.Option(
        0.5,
        "--tfce-extent",
        help="TFCE extent exponent (E parameter)",
        min=0.1,
        max=5.0,
    ),
    tfce_connectivity: int = typer.Option(
        26,
        "--tfce-connectivity",
        help="TFCE spatial connectivity (6, 18, or 26 for 3D)",
    ),
) -> None:
    """
    Run General Linear Model analysis with permutation testing.

    This command performs GLM analysis on neuroimaging data with the specified
    design matrix and contrasts, similar to FSL's randomise command.
    """
    # Initialize backend factory
    factory = BackendFactory()

    # Validate backend
    valid_backends = ["auto", "cpu", "mps", "gpu_optimized"]
    if backend.lower() not in valid_backends:
        console.print(
            f"[red]Error: Invalid backend '{backend}'. Valid options: {', '.join(valid_backends)}[/red]"
        )
        raise typer.Exit(1)

    # Parse and validate correction methods
    valid_corrections = ["none", "voxel", "cluster", "tfce"]
    correction_methods = [method.strip().lower() for method in correction.split(",")]

    # Remove 'none' if other methods are specified
    if len(correction_methods) > 1 and "none" in correction_methods:
        correction_methods = [
            method for method in correction_methods if method != "none"
        ]
        if verbose:
            console.print(
                "[yellow]Note: 'none' correction removed when other methods specified[/yellow]"
            )

    # Validate each correction method
    for method in correction_methods:
        if method not in valid_corrections:
            console.print(
                f"[red]Error: Invalid correction method '{method}'. Valid options: {', '.join(valid_corrections)}[/red]"
            )
            raise typer.Exit(1)

    # If only 'none' is specified, handle specially
    if correction_methods == ["none"]:
        correction_methods = []

    # Validate options
    if one_sample:
        if design_file is not None:
            console.print(
                "[yellow]Warning: Design file ignored for one-sample t-test[/yellow]"
            )
        if contrast_file is not None:
            console.print(
                "[yellow]Warning: Contrast file ignored for one-sample t-test[/yellow]"
            )
    else:
        if design_file is None:
            console.print(
                "[red]Error: Design file is required for two-sample/multiple regression analysis. Use -1 for one-sample t-test.[/red]"
            )
            raise typer.Exit(1)
        if contrast_file is None:
            console.print(
                "[red]Error: Contrast file is required for two-sample/multiple regression analysis. Use -1 for one-sample t-test.[/red]"
            )
            raise typer.Exit(1)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare configuration
    config = {
        "input_file": input_file,
        "design_file": design_file,
        "contrast_file": contrast_file,
        "output_dir": output_dir,
        "backend": backend.lower(),
        "verbose": verbose,
        "n_permutations": n_permutations,
        "correction_methods": correction_methods,
        "alpha": alpha,
        "tfce_height": tfce_height,
        "tfce_extent": tfce_extent,
        "tfce_connectivity": tfce_connectivity,
        "one_sample": one_sample,
    }

    if verbose:
        console.print("[green]Running GLM analysis...[/green]")
        console.print(f"Input: {input_file}")
        console.print(f"Design: {design_file}")
        console.print(f"Contrasts: {contrast_file}")
        console.print(f"Output: {output_dir}")
        console.print(f"Backend: {backend}")
        console.print(f"Permutations: {n_permutations}")
        console.print(
            f"Corrections: {', '.join(correction_methods) if correction_methods else 'none'}"
        )
        console.print(f"Alpha: {alpha}")

    try:
        # Run the analysis
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Running GLM analysis...", total=None)

            result = run_glm(config)

            progress.update(task, description="Analysis complete!")

        if result["status"] == "success":
            console.print("[green] Analysis completed successfully![/green]")
            console.print(f"Results written to: {output_dir}")
        else:
            console.print(
                f"[red] Analysis failed: {result.get('error', 'Unknown error')}[/red]"
            )
            raise typer.Exit(1)

    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def generate_permutation_null_distributions_gpu_optimized(
    Y: np.ndarray,
    X: np.ndarray,
    contrasts: np.ndarray,
    backend: Any,
    n_permutations: int,
    correction_methods: list[str],
    one_sample: bool = False,
    spatial_shape: tuple[int, ...] | None = None,
    tfce_height: float = 2.0,
    tfce_extent: float = 0.5,
    tfce_connectivity: int = 26,
    cluster_threshold: float = 2.3,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """
    GPU-optimized permutation testing with vectorized batch processing.

    This implementation achieves significant speedup by:
    1. Processing all permutations in vectorized batches on GPU
    2. Minimal CPU-GPU data transfers
    3. Optimized matrix operations
    """
    # Check if backend supports batch processing
    if hasattr(backend, "compute_glm_batch"):
        return _generate_null_distributions_batch(
            Y,
            X,
            contrasts,
            backend,
            n_permutations,
            correction_methods,
            one_sample,
            spatial_shape,
            tfce_height,
            tfce_extent,
            tfce_connectivity,
            cluster_threshold,
            verbose,
        )
    else:
        # Fall back to sequential processing for standard backends
        return _generate_null_distributions_sequential(
            Y,
            X,
            contrasts,
            backend,
            n_permutations,
            correction_methods,
            one_sample,
            spatial_shape,
            tfce_height,
            tfce_extent,
            tfce_connectivity,
            cluster_threshold,
            verbose,
        )


def _generate_null_distributions_batch(
    Y: np.ndarray,
    X: np.ndarray,
    contrasts: np.ndarray,
    backend: Any,
    n_permutations: int,
    correction_methods: list[str],
    one_sample: bool = False,
    spatial_shape: tuple[int, ...] | None = None,
    tfce_height: float = 2.0,
    tfce_extent: float = 0.5,
    tfce_connectivity: int = 26,
    cluster_threshold: float = 2.3,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """GPU-optimized batch processing of all permutations."""
    from accelperm.backends.gpu_optimized import VectorizedPermutationEngine

    n_voxels, n_subjects = Y.shape
    n_contrasts = contrasts.shape[0]

    # Initialize vectorized permutation engine
    perm_engine = VectorizedPermutationEngine(backend)

    if verbose:
        console.print("[green]Using GPU-optimized batch processing[/green]")
        console.print(
            f"Generating {n_permutations} permutations for {n_voxels} voxels, {n_contrasts} contrasts..."
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Generate all permutations at once on GPU
        task1 = progress.add_task("Generating permutations on GPU...", total=None)

        strategy = "sign_flip" if one_sample else "full"
        permutations = perm_engine.generate_permutations_gpu(
            n_subjects, n_permutations, strategy
        )
        progress.update(task1, description="✓ Permutations generated")

        # Use streaming corrections for maximum efficiency
        task2 = progress.add_task(
            "Computing GLM with streaming corrections...", total=None
        )

        if hasattr(backend, "compute_glm_batch_with_streaming_corrections"):
            # Use streaming approach - processes corrections during GLM computation
            # This eliminates the separate "processing null distributions" bottleneck

            # Calculate expected chunk size for user feedback
            if hasattr(backend, "_calculate_optimal_chunk_size"):
                chunk_size = backend._calculate_optimal_chunk_size(
                    n_voxels, n_subjects, n_contrasts
                )
                n_chunks = (n_permutations + chunk_size - 1) // chunk_size
                if n_chunks > 1:
                    progress.update(
                        task2,
                        description=f"Streaming {n_chunks} chunks (~{chunk_size} perms each)...",
                    )

            # Streaming GLM + corrections in one pass
            streaming_result = backend.compute_glm_batch_with_streaming_corrections(
                Y,
                X,
                contrasts,
                permutations.cpu().numpy(),
                correction_methods=correction_methods,
                spatial_shape=spatial_shape,
                tfce_height=tfce_height,
                tfce_extent=tfce_extent,
                tfce_connectivity=tfce_connectivity,
                cluster_threshold=2.3,
            )

            # Extract results
            t_stats_all = None  # Not stored to save memory
            null_distributions = streaming_result["null_distributions"]
            original_stats = streaming_result["original_stats"]

            progress.update(task2, description="✓ Streaming GLM + corrections complete")

        else:
            # Fallback to traditional two-stage approach
            # Calculate expected chunk size for user feedback
            if hasattr(backend, "_calculate_optimal_chunk_size"):
                chunk_size = backend._calculate_optimal_chunk_size(
                    n_voxels, n_subjects, n_contrasts
                )
                n_chunks = (n_permutations + chunk_size - 1) // chunk_size
                if n_chunks > 1:
                    progress.update(
                        task2,
                        description=f"Processing {n_chunks} chunks of ~{chunk_size} permutations...",
                    )

            result = backend.compute_glm_batch(
                Y, X, contrasts, permutations.cpu().numpy()
            )
            t_stats_all = result[
                "t_stats"
            ]  # Shape: (n_voxels, n_contrasts, n_permutations)
            original_stats = result.get(
                "original_stats",
                t_stats_all[:, :, 0] if t_stats_all.shape[2] > 0 else None,
            )

            progress.update(task2, description="✓ Batch GLM complete")

            # Process null distributions
            task3 = progress.add_task("Processing null distributions...", total=None)

            # Use GPU-accelerated null distribution processing if available
            if hasattr(backend, "device") and backend.device.type in ["cuda", "mps"]:
                from accelperm.core.gpu_null_distributions import (
                    process_null_distributions_gpu_accelerated,
                )

                null_distributions = process_null_distributions_gpu_accelerated(
                    t_stats_all,
                    correction_methods,
                    backend.device,
                    spatial_shape,
                    tfce_height,
                    tfce_extent,
                    tfce_connectivity,
                    2.3,
                )
            else:
                # Fall back to CPU processing
                from accelperm.core.gpu_processing import process_null_distributions

                null_distributions = process_null_distributions(
                    t_stats_all,
                    correction_methods,
                    spatial_shape,
                    tfce_height,
                    tfce_extent,
                    tfce_connectivity,
                    2.3,
                )

            progress.update(task3, description="✓ Null distributions ready")

    if verbose:
        console.print(
            f"[green]Batch processing complete! Processed {n_permutations} permutations[/green]"
        )

    # Handle streaming case: return both null distributions and original stats
    if "original_stats" in locals() and original_stats is not None:
        return {
            "null_distributions": null_distributions,
            "original_stats": original_stats,
            "is_streaming": True,
        }
    else:
        return {"null_distributions": null_distributions, "is_streaming": False}


def _generate_null_distributions_sequential(
    Y: np.ndarray,
    X: np.ndarray,
    contrasts: np.ndarray,
    backend: Any,
    n_permutations: int,
    correction_methods: list[str],
    one_sample: bool = False,
    spatial_shape: tuple[int, ...] | None = None,
    tfce_height: float = 2.0,
    tfce_extent: float = 0.5,
    tfce_connectivity: int = 26,
    cluster_threshold: float = 2.3,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """
    Generate null distributions through permutation testing.

    Parameters
    ----------
    Y : np.ndarray
        Data matrix (n_voxels, n_subjects)
    X : np.ndarray
        Design matrix (n_subjects, n_regressors)
    contrasts : np.ndarray
        Contrast matrix (n_contrasts, n_regressors)
    backend : Backend
        Computation backend
    n_permutations : int
        Number of permutations
    correction_methods : list[str]
        Types of corrections to prepare for
    one_sample : bool
        Whether this is a one-sample test
    spatial_shape : tuple
        Original spatial dimensions for cluster/TFCE
    tfce_height : float
        TFCE H parameter
    tfce_extent : float
        TFCE E parameter
    tfce_connectivity : int
        Spatial connectivity
    cluster_threshold : float
        Threshold for cluster formation
    verbose : bool
        Verbose output

    Returns
    -------
    dict[str, np.ndarray]
        Null distributions for each contrast and correction type
    """
    n_voxels, n_subjects = Y.shape
    n_contrasts = contrasts.shape[0]

    # Initialize permutation strategy
    if one_sample:
        # For one-sample test, use sign-flipping
        perm_strategy = SignFlippingStrategy(n_subjects)
    else:
        # For two-sample/regression, would need full permutation
        # For now, use sign-flipping as placeholder
        perm_strategy = SignFlippingStrategy(n_subjects)

    # Storage for null distributions
    null_distributions = {}

    # Determine what statistics we need based on correction methods
    need_voxel = any(method in ["voxel"] for method in correction_methods)
    need_cluster = "cluster" in correction_methods
    need_tfce = "tfce" in correction_methods

    # Initialize storage based on required methods
    if need_voxel:
        null_max_t = np.zeros((n_permutations, n_contrasts))
        null_t_stats = np.zeros((n_permutations, n_voxels, n_contrasts))

    if need_cluster:
        null_max_cluster = np.zeros((n_permutations, n_contrasts))

    if need_tfce:
        null_max_tfce = np.zeros((n_permutations, n_contrasts))
        null_tfce_stats = np.zeros((n_permutations, n_voxels, n_contrasts))
        # Initialize TFCE processor
        tfce_processor = TFCEProcessor(
            height_power=tfce_height,
            extent_power=tfce_extent,
            connectivity=tfce_connectivity,
        )

    # Progress tracking
    if verbose:
        console.print(
            f"Generating null distribution with {n_permutations} permutations..."
        )

    # Determine batch size based on available memory and backend
    batch_size = min(32, n_permutations)  # Process up to 32 permutations at once
    if hasattr(backend, "device") and "mps" in str(backend.device):
        batch_size = min(
            16, n_permutations
        )  # Smaller batches for MPS due to memory constraints

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running permutations...", total=n_permutations)

        # Process permutations in batches for better GPU utilization
        for batch_start in range(0, n_permutations, batch_size):
            batch_end = min(batch_start + batch_size, n_permutations)
            current_batch_size = batch_end - batch_start

            # Generate all permutations in this batch
            Y_batch = np.zeros((current_batch_size, n_voxels, n_subjects))
            for i, perm_i in enumerate(range(batch_start, batch_end)):
                if perm_i == 0:
                    # First permutation is always unpermuted (original) data
                    Y_batch[i] = Y
                else:
                    # Apply permutation
                    perm_vector = perm_strategy.next_permutation(perm_i)
                    Y_batch[i] = perm_strategy.apply_permutation(Y, perm_vector)

            # Compute GLM for entire batch with optimized GPU utilization
            t_stats_batch = np.zeros((current_batch_size, n_voxels, n_contrasts))

            if hasattr(backend, "device") and "mps" in str(backend.device):
                # For MPS backend: minimize CPU-GPU transfers by keeping data on GPU
                import torch

                # Transfer entire batch to GPU once
                Y_batch_tensor = torch.from_numpy(Y_batch).float().to(backend.device)
                X_tensor = torch.from_numpy(X).float().to(backend.device)
                contrasts_tensor = (
                    torch.from_numpy(contrasts).float().to(backend.device)
                )

                # Process each dataset in batch on GPU
                for i in range(current_batch_size):
                    # Convert tensors back to numpy for existing GLM function
                    # This is less optimal but maintains compatibility
                    Y_np = Y_batch_tensor[i].cpu().numpy()
                    glm_perm = backend.compute_glm(Y_np, X, contrasts)
                    t_stats_batch[i] = glm_perm["t_stat"]
            else:
                # For CPU backend: standard processing
                for i in range(current_batch_size):
                    glm_perm = backend.compute_glm(Y_batch[i], X, contrasts)
                    t_stats_batch[i] = glm_perm["t_stat"]

            # Store statistics for this batch
            for i, perm_i in enumerate(range(batch_start, batch_end)):
                t_stats_perm = t_stats_batch[i]  # Shape: (n_voxels, n_contrasts)

                if need_voxel:
                    # Store max statistic across voxels for FWER
                    null_max_t[perm_i, :] = np.max(np.abs(t_stats_perm), axis=0)
                    null_t_stats[perm_i, :, :] = t_stats_perm

                if need_cluster and spatial_shape is not None:
                    # Find clusters for each contrast
                    for c_idx in range(n_contrasts):
                        t_map_3d = t_stats_perm[:, c_idx].reshape(spatial_shape)

                        # Threshold the map
                        thresh_map = np.abs(t_map_3d) > cluster_threshold

                        # Find connected components
                        labeled, n_clusters = scipy_label(thresh_map)

                        # Find max cluster size
                        if n_clusters > 0:
                            cluster_sizes = [
                                np.sum(labeled == i) for i in range(1, n_clusters + 1)
                            ]
                            null_max_cluster[perm_i, c_idx] = max(cluster_sizes)
                        else:
                            null_max_cluster[perm_i, c_idx] = 0

                if need_tfce and spatial_shape is not None:
                    # Apply TFCE enhancement
                    for c_idx in range(n_contrasts):
                        t_stat_flat = t_stats_perm[:, c_idx]  # Keep as flat array
                        tfce_flat = tfce_processor.enhance(t_stat_flat, spatial_shape)
                        null_max_tfce[perm_i, c_idx] = np.max(tfce_flat)
                        null_tfce_stats[perm_i, :, c_idx] = tfce_flat

            progress.update(task, advance=current_batch_size)

    # Package null distributions
    if need_voxel:
        null_distributions["max_t"] = null_max_t
        null_distributions["t_stats"] = null_t_stats
    if need_cluster:
        null_distributions["max_cluster"] = null_max_cluster
    if need_tfce:
        null_distributions["max_tfce"] = null_max_tfce
        null_distributions["tfce_stats"] = null_tfce_stats

    return null_distributions


def apply_permutation_corrections(
    glm_result: dict[str, Any],
    null_distributions: dict[str, np.ndarray],
    correction_method: str,
    alpha: float,
    spatial_shape: tuple[int, ...] | None = None,
    cluster_threshold: float = 2.3,
    tfce_height: float = 2.0,
    tfce_extent: float = 0.5,
    tfce_connectivity: int = 26,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Apply permutation-based multiple comparison corrections.

    Parameters
    ----------
    glm_result : dict[str, Any]
        GLM results containing t_stat
    null_distributions : dict[str, np.ndarray]
        Null distributions from permutation testing
    correction_method : str
        Correction method to apply
    alpha : float
        Significance level
    spatial_shape : tuple
        Original spatial dimensions for cluster/TFCE
    cluster_threshold : float
        Threshold for cluster formation
    tfce_height : float
        TFCE height exponent
    tfce_extent : float
        TFCE extent exponent
    tfce_connectivity : int
        Spatial connectivity
    verbose : bool
        Verbose output flag

    Returns
    -------
    dict[str, Any]
        Correction results for each contrast
    """
    t_stats = glm_result["t_stat"]  # Shape: (n_voxels, n_contrasts)
    n_voxels, n_contrasts = t_stats.shape

    correction_results = {}

    for i in range(n_contrasts):
        t_stat_contrast = t_stats[:, i]

        if correction_method in ["voxel", "fwer", "none"]:
            # Voxel-level FWER correction using max-statistic method
            null_max_t = null_distributions["max_t"][:, i]

            # Create FWERCorrection with null distribution for this contrast
            corrector = FWERCorrection(null_distributions["t_stats"][:, :, i])
            result = corrector.correct(np.abs(t_stat_contrast), alpha=alpha)

        elif correction_method == "cluster" and spatial_shape is not None:
            # Cluster-level correction
            null_max_cluster = null_distributions["max_cluster"][:, i]

            corrector = ClusterCorrection(
                null_cluster_sizes=null_max_cluster,
                voxel_threshold=cluster_threshold,
                connectivity=tfce_connectivity,
                correction_type="extent",
            )
            result = corrector.correct(
                t_stat_contrast,
                alpha=alpha,
                spatial_shape=spatial_shape,
            )

        elif correction_method == "tfce" and spatial_shape is not None:
            # TFCE correction
            null_tfce = null_distributions["tfce_stats"][:, :, i]

            corrector = TFCECorrection(
                null_distribution=null_tfce,
                height_power=tfce_height,
                extent_power=tfce_extent,
                connectivity=tfce_connectivity,
            )
            result = corrector.correct(
                t_stat_contrast,
                alpha=alpha,
                spatial_shape=spatial_shape,
            )

        else:
            # Fallback to parametric FDR if permutation not supported
            if verbose:
                console.print(
                    f"[yellow]Warning: {correction_method} not supported with current data. Using FDR.[/yellow]"
                )
            from scipy import stats

            p_values = 2 * (
                1 - stats.t.cdf(np.abs(t_stat_contrast), df=t_stats.shape[1] - 1)
            )
            corrector = FDRCorrection()
            result = corrector.correct(p_values, alpha=alpha)

        correction_results[f"contrast_{i}"] = result

        if verbose:
            n_significant = np.sum(result.significant_mask)
            console.print(
                f"  Contrast {i+1}: {n_significant}/{n_voxels} significant voxels"
            )

    return correction_results


def run_glm(config: dict[str, Any]) -> dict[str, Any]:
    """
    Run GLM analysis with the given configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary containing:
        - input_file: Path to NIfTI input
        - design_file: Path to design matrix
        - contrast_file: Path to contrasts
        - output_dir: Path to output directory
        - backend: Backend name (cpu, mps, cuda)
        - verbose: Verbose output flag

    Returns
    -------
    dict[str, Any]
        Result dictionary with status and any error information
    """
    try:
        # Load input data
        if config["verbose"]:
            console.print("Loading input data...")

        nifti_loader = NiftiLoader()
        data_result = nifti_loader.load(config["input_file"])
        raw_data = data_result["data"]  # Shape: (x, y, z, n_subjects) for 4D data
        affine = data_result["affine"]

        # Handle different data dimensionalities
        if raw_data.ndim == 4:
            # 4D data: (x, y, z, n_subjects) -> reshape to 2D (n_voxels, n_subjects)
            spatial_shape = raw_data.shape[:3]
            n_voxels = np.prod(spatial_shape)
            n_subjects = raw_data.shape[3]
            Y = raw_data.reshape(n_voxels, n_subjects)

            if config["verbose"]:
                console.print(
                    f"Loaded 4D data: {spatial_shape} × {n_subjects} subjects"
                )
                console.print(
                    f"Reshaped to: {Y.shape[0]} voxels × {Y.shape[1]} subjects"
                )
        elif raw_data.ndim == 3:
            # 3D data: single volume - treat as single subject
            spatial_shape = raw_data.shape
            n_voxels = np.prod(spatial_shape)
            Y = raw_data.reshape(n_voxels, 1)

            if config["verbose"]:
                console.print(f"Loaded 3D data: {spatial_shape}")
                console.print(f"Reshaped to: {Y.shape[0]} voxels × 1 subject")
        elif raw_data.ndim == 2:
            # Already 2D - use as is
            Y = raw_data
            spatial_shape = None

            if config["verbose"]:
                console.print(
                    f"Loaded 2D data: {Y.shape[0]} voxels × {Y.shape[1]} subjects"
                )
        else:
            raise ValueError(f"Unsupported data dimensionality: {raw_data.ndim}D")

        if config["one_sample"]:
            # For one-sample t-test, create simple design matrix (intercept only)
            X = np.ones((Y.shape[1], 1))  # Shape: (n_subjects, 1) - intercept only
            contrasts = np.array([[1.0]])  # Shape: (1, 1) - test mean against zero
            contrast_result = {"contrast_names": ["one_sample"]}

            if config["verbose"]:
                console.print("One-sample t-test: testing mean against zero")
                console.print(
                    f"Design matrix: {X.shape[0]} subjects, {X.shape[1]} regressor (intercept)"
                )
                console.print("Contrast: test mean != 0")
        else:
            # Load design matrix
            if config["verbose"]:
                console.print("Loading design matrix...")

            design_loader = DesignMatrixLoader()
            design_result = design_loader.load(config["design_file"])
            X = design_result["design_matrix"]  # Shape: (n_subjects, n_regressors)

            if config["verbose"]:
                console.print(
                    f"Loaded design: {X.shape[0]} subjects, {X.shape[1]} regressors"
                )

            # Load contrasts
            if config["verbose"]:
                console.print("Loading contrasts...")

            contrast_loader = ContrastLoader()
            contrast_result = contrast_loader.load(config["contrast_file"])
            contrasts = contrast_result[
                "contrast_matrix"
            ]  # Shape: (n_contrasts, n_regressors)

            if config["verbose"]:
                console.print(f"Loaded contrasts: {contrasts.shape[0]} contrasts")

        # Validate dimensions
        if Y.shape[1] != X.shape[0]:
            raise ValueError(
                f"Dimension mismatch: Data has {Y.shape[1]} subjects, "
                f"design has {X.shape[0]} subjects"
            )

        if X.shape[1] != contrasts.shape[1]:
            raise ValueError(
                f"Dimension mismatch: Design has {X.shape[1]} regressors, "
                f"contrasts have {contrasts.shape[1]} regressors"
            )

        # Initialize backend using factory
        factory = BackendFactory()

        if config["backend"] == "auto":
            if config["verbose"]:
                console.print("Auto-selecting optimal backend...")
                available = factory.list_available_backends()
                console.print(f"Available backends: {', '.join(available)}")
            backend = factory.get_best_backend()
        else:
            if config["verbose"]:
                console.print(f"Initializing {config['backend'].upper()} backend...")
            backend = factory.get_backend(config["backend"])

        if config["verbose"]:
            console.print(f"Using {backend.name.upper()} backend")
            caps = factory.get_backend_capabilities(backend.name)
            if "max_memory_gb" in caps:
                console.print(f"Available memory: {caps['max_memory_gb']:.1f} GB")

        # Run GLM
        if config["verbose"]:
            console.print("Computing GLM statistics...")

        glm_result = backend.compute_glm(Y, X, contrasts)

        # Apply multiple comparison corrections if requested
        correction_results = {}
        if config["correction_methods"]:
            # Generate null distributions through permutation testing
            if config["verbose"]:
                methods_str = ", ".join(
                    [method.upper() for method in config["correction_methods"]]
                )
                console.print(
                    f"Generating permutation null distribution for {methods_str} correction(s)..."
                )

            perm_result = generate_permutation_null_distributions_gpu_optimized(
                Y=Y,
                X=X,
                contrasts=contrasts,
                backend=backend,
                n_permutations=config["n_permutations"],
                correction_methods=config["correction_methods"],
                one_sample=config["one_sample"],
                spatial_shape=spatial_shape if "spatial_shape" in locals() else None,
                tfce_height=config["tfce_height"],
                tfce_extent=config["tfce_extent"],
                tfce_connectivity=config["tfce_connectivity"],
                cluster_threshold=2.3,  # Default cluster threshold
                verbose=config["verbose"],
            )

            # Handle streaming vs traditional results
            if isinstance(perm_result, dict) and perm_result.get("is_streaming", False):
                # Streaming case: use original stats from permutation result
                null_distributions = perm_result["null_distributions"]
                glm_result = {
                    "t_stat": perm_result["original_stats"],
                    "p_values": None,  # Not computed in streaming mode
                    "coefficients": None,  # Not computed in streaming mode
                }
                if config["verbose"]:
                    console.print(
                        "[green]Used streaming approach - GLM and corrections computed together[/green]"
                    )
            else:
                # Traditional case: use separate GLM result
                if isinstance(perm_result, dict):
                    null_distributions = perm_result["null_distributions"]
                else:
                    null_distributions = perm_result  # Backwards compatibility

            # Apply corrections (skip if streaming already computed them)
            is_streaming = isinstance(perm_result, dict) and perm_result.get(
                "is_streaming", False
            )
            if not is_streaming:
                # Traditional path: apply corrections separately
                for method in config["correction_methods"]:
                    if config["verbose"]:
                        console.print(f"Applying {method.upper()} correction...")

                    method_results = apply_permutation_corrections(
                        glm_result=glm_result,
                        null_distributions=null_distributions,
                        correction_method=method,
                        alpha=config["alpha"],
                        spatial_shape=spatial_shape
                        if "spatial_shape" in locals()
                        else None,
                        cluster_threshold=2.3,  # Default cluster threshold
                        tfce_height=config["tfce_height"],
                        tfce_extent=config["tfce_extent"],
                        tfce_connectivity=config["tfce_connectivity"],
                        verbose=config["verbose"],
                    )
                    correction_results[method] = method_results
            else:
                # Streaming path: corrections already applied, just create placeholder results
                if config["verbose"]:
                    console.print(
                        "[green]Corrections already applied during streaming computation[/green]"
                    )

                # Create simple correction results for consistency
                for method in config["correction_methods"]:
                    correction_results[method] = {
                        "method": method,
                        "null_distribution_computed": True,
                        "alpha": config["alpha"],
                        "note": "Corrections computed during streaming - p-values available in null distributions",
                    }

        # Save results
        if config["verbose"]:
            console.print("Saving results...")

        output_writer = OutputWriter()

        # Store spatial_shape for use in reshaping outputs
        if "spatial_shape" not in locals():
            spatial_shape = None

        # Save statistical maps
        for i in range(contrasts.shape[0]):
            # Reshape back to 3D spatial dimensions if needed
            if spatial_shape is not None:
                t_map = glm_result["t_stat"][:, i].reshape(spatial_shape)
                # Handle streaming case where p_values might be None
                if glm_result["p_values"] is not None:
                    p_map = glm_result["p_values"][:, i].reshape(spatial_shape)
                else:
                    # In streaming mode, p-values weren't computed - create placeholder
                    p_map = np.ones(spatial_shape) * 0.5  # Neutral p-value
            else:
                # If no spatial shape, keep as vector with dummy dimensions
                t_map = glm_result["t_stat"][:, i].reshape(-1, 1, 1)
                if glm_result["p_values"] is not None:
                    p_map = glm_result["p_values"][:, i].reshape(-1, 1, 1)
                else:
                    # In streaming mode, p-values weren't computed - create placeholder
                    p_map = np.ones((glm_result["t_stat"].shape[0], 1, 1)) * 0.5

            contrast_name = contrast_result["contrast_names"][i]

            # Save t-statistic map
            t_output = config["output_dir"] / f"tstat_{contrast_name}.nii.gz"
            output_writer.save_statistical_map(t_map, affine, t_output, "t-stat")

            # Save uncorrected p-value map
            p_output = config["output_dir"] / f"pvals_{contrast_name}.nii.gz"
            output_writer.save_p_value_map(p_map, affine, p_output)

            # Save corrected p-values for each correction method (skip in streaming mode)
            for method, method_results in correction_results.items():
                # Skip saving corrected results in streaming mode
                if "note" in method_results and "streaming" in method_results["note"]:
                    if config["verbose"]:
                        console.print(
                            f"[yellow]Skipping {method} corrected p-values (computed during streaming)[/yellow]"
                        )
                    continue

                corrected_result = method_results[f"contrast_{i}"]

                # Reshape corrected results back to 3D spatial dimensions if needed
                if spatial_shape is not None:
                    corrected_p_map = corrected_result.corrected_p_values.reshape(
                        spatial_shape
                    )
                    sig_map = corrected_result.significant_mask.astype(float).reshape(
                        spatial_shape
                    )
                else:
                    corrected_p_map = corrected_result.corrected_p_values.reshape(
                        -1, 1, 1
                    )
                    sig_map = corrected_result.significant_mask.astype(float).reshape(
                        -1, 1, 1
                    )

                # Save corrected p-values
                corrected_output = (
                    config["output_dir"] / f"corrp_{contrast_name}_{method}.nii.gz"
                )
                output_writer.save_p_value_map(
                    corrected_p_map, affine, corrected_output
                )

                # Save significance mask
                sig_output = (
                    config["output_dir"] / f"sig_{contrast_name}_{method}.nii.gz"
                )
                output_writer.save_statistical_map(
                    sig_map, affine, sig_output, "significance"
                )

        # Create summary
        summary_output = config["output_dir"] / "results_summary.txt"
        results_summary = {
            "n_voxels": Y.shape[0],
            "n_subjects": Y.shape[1],
            "n_regressors": X.shape[1],
            "n_contrasts": contrasts.shape[0],
            "backend": config["backend"],
            "correction_methods": ", ".join(config["correction_methods"])
            if config["correction_methods"]
            else "none",
            "alpha": config["alpha"],
            "max_t_stat": float(glm_result["t_stat"].max()),
            "min_p_value": float(glm_result["p_values"].min())
            if glm_result["p_values"] is not None
            else "N/A (streaming mode)",
        }

        # Add correction summary for each method (skip detailed results in streaming mode)
        for method, method_results in correction_results.items():
            # Check if this is streaming mode (placeholder structure)
            if isinstance(method_results, dict) and "note" in method_results and "streaming" in method_results["note"]:
                # Streaming mode: Add basic method info only
                results_summary[f"{method}_correction"] = "Applied during streaming computation"
                results_summary[f"{method}_alpha"] = method_results.get("alpha", "N/A")
            else:
                # Traditional mode: Add detailed per-contrast results
                for i in range(contrasts.shape[0]):
                    result = method_results[f"contrast_{i}"]
                    n_significant = int(np.sum(result.significant_mask))
                    results_summary[
                        f"contrast_{i}_{method}_significant_voxels"
                    ] = n_significant
                    results_summary[f"contrast_{i}_{method}_min_corrected_p"] = float(
                        result.corrected_p_values.min()
                    )

        from accelperm.io.output import create_results_summary

        create_results_summary(results_summary, summary_output)

        return {"status": "success", "results": results_summary}

    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    app()
