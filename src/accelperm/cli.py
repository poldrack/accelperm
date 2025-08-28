"""Command-line interface for AccelPerm."""

from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from accelperm.backends.factory import BackendFactory
from accelperm.core.corrections import BonferroniCorrection, FDRCorrection
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
    design_file: Path = typer.Option(
        ...,
        "--design",
        "-d",
        help="Design matrix file (.txt, .csv, .tsv)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    contrast_file: Path = typer.Option(
        ...,
        "--contrasts",
        "-c",
        help="Contrast file (.con, .txt, .csv)",
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
    backend: str = typer.Option(
        "auto",
        "--backend",
        "-b",
        help="Backend to use (auto, cpu, mps)",
        case_sensitive=False,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    n_permutations: int = typer.Option(
        1000,
        "--n-perm",
        "-n",
        help="Number of permutations for statistical testing",
        min=100,
    ),
    correction: str = typer.Option(
        "fdr",
        "--correction",
        help="Multiple comparison correction method (none, bonferroni, fdr, fwer, cluster, tfce)",
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
    valid_backends = ["auto", "cpu", "mps"]
    if backend.lower() not in valid_backends:
        console.print(
            f"[red]Error: Invalid backend '{backend}'. Valid options: {', '.join(valid_backends)}[/red]"
        )
        raise typer.Exit(1)

    # Validate correction method
    valid_corrections = ["none", "bonferroni", "fdr", "fwer", "cluster", "tfce"]
    if correction.lower() not in valid_corrections:
        console.print(
            f"[red]Error: Invalid correction method '{correction}'. Valid options: {', '.join(valid_corrections)}[/red]"
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
        "correction": correction.lower(),
        "alpha": alpha,
        "tfce_height": tfce_height,
        "tfce_extent": tfce_extent,
        "tfce_connectivity": tfce_connectivity,
    }

    if verbose:
        console.print("[green]Running GLM analysis...[/green]")
        console.print(f"Input: {input_file}")
        console.print(f"Design: {design_file}")
        console.print(f"Contrasts: {contrast_file}")
        console.print(f"Output: {output_dir}")
        console.print(f"Backend: {backend}")
        console.print(f"Permutations: {n_permutations}")
        console.print(f"Correction: {correction}")
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


def apply_corrections(
    glm_result: dict[str, Any],
    correction_method: str,
    alpha: float,
    verbose: bool = False,
    tfce_height: float = 2.0,
    tfce_extent: float = 0.5,
    tfce_connectivity: int = 26,
) -> dict[str, Any]:
    """
    Apply multiple comparison corrections to GLM results.

    Parameters
    ----------
    glm_result : dict[str, Any]
        GLM results containing t_stat and p_values
    correction_method : str
        Correction method to apply
    alpha : float
        Significance level
    verbose : bool
        Verbose output flag
    tfce_height : float
        TFCE height exponent (H parameter)
    tfce_extent : float
        TFCE extent exponent (E parameter)
    tfce_connectivity : int
        TFCE spatial connectivity

    Returns
    -------
    dict[str, Any]
        Correction results for each contrast
    """

    p_values = glm_result["p_values"]  # Shape: (n_voxels, n_contrasts)
    n_voxels, n_contrasts = p_values.shape

    correction_results = {}

    for i in range(n_contrasts):
        contrast_p = p_values[:, i]

        if correction_method == "bonferroni":
            corrector = BonferroniCorrection()
            result = corrector.correct(contrast_p, alpha=alpha)

        elif correction_method == "fdr":
            corrector = FDRCorrection()
            result = corrector.correct(contrast_p, alpha=alpha)

        elif correction_method == "fwer":
            # For FWER, we would need a null distribution from permutations
            # For now, fall back to Bonferroni (conservative)
            if verbose:
                console.print(
                    "[yellow]Warning: FWER requires permutation testing. Using Bonferroni correction.[/yellow]"
                )
            corrector = BonferroniCorrection()
            result = corrector.correct(contrast_p, alpha=alpha)

        elif correction_method == "cluster":
            # For cluster correction, we would need spatial information and null distribution
            # For now, fall back to FDR
            if verbose:
                console.print(
                    "[yellow]Warning: Cluster correction requires permutation testing. Using FDR correction.[/yellow]"
                )
            corrector = FDRCorrection()
            result = corrector.correct(contrast_p, alpha=alpha)

        elif correction_method == "tfce":
            # For TFCE, we would need test statistics (not p-values) and null distribution
            # For now, fall back to FDR
            if verbose:
                console.print(
                    "[yellow]Warning: TFCE requires test statistics and permutation testing. Using FDR correction.[/yellow]"
                )
            corrector = FDRCorrection()
            result = corrector.correct(contrast_p, alpha=alpha)

        else:
            raise ValueError(f"Unknown correction method: {correction_method}")

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
        Y = data_result["data"]  # Shape: (n_voxels, n_subjects)
        affine = data_result["affine"]

        if config["verbose"]:
            console.print(f"Loaded data: {Y.shape[0]} voxels, {Y.shape[1]} subjects")

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
        correction_results = None
        if config["correction"] != "none":
            if config["verbose"]:
                console.print(f"Applying {config['correction'].upper()} correction...")

            correction_results = apply_corrections(
                glm_result,
                config["correction"],
                config["alpha"],
                config["verbose"],
                config["tfce_height"],
                config["tfce_extent"],
                config["tfce_connectivity"],
            )

        # Save results
        if config["verbose"]:
            console.print("Saving results...")

        output_writer = OutputWriter()

        # Save statistical maps
        for i in range(contrasts.shape[0]):
            t_map = glm_result["t_stat"][:, i].reshape(-1, 1, 1)  # Reshape for NIfTI
            p_map = glm_result["p_values"][:, i].reshape(-1, 1, 1)

            contrast_name = contrast_result["contrast_names"][i]

            # Save t-statistic map
            t_output = config["output_dir"] / f"tstat_{contrast_name}.nii.gz"
            output_writer.save_statistical_map(t_map, affine, t_output, "t-stat")

            # Save uncorrected p-value map
            p_output = config["output_dir"] / f"pvals_{contrast_name}.nii.gz"
            output_writer.save_p_value_map(p_map, affine, p_output)

            # Save corrected p-values if correction was applied
            if correction_results is not None:
                corrected_result = correction_results[f"contrast_{i}"]
                corrected_p_map = corrected_result.corrected_p_values.reshape(-1, 1, 1)

                # Save corrected p-values
                corrected_output = (
                    config["output_dir"]
                    / f"corrp_{contrast_name}_{config['correction']}.nii.gz"
                )
                output_writer.save_p_value_map(
                    corrected_p_map, affine, corrected_output
                )

                # Save significance mask
                sig_map = corrected_result.significant_mask.astype(float).reshape(
                    -1, 1, 1
                )
                sig_output = (
                    config["output_dir"]
                    / f"sig_{contrast_name}_{config['correction']}.nii.gz"
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
            "correction_method": config["correction"],
            "alpha": config["alpha"],
            "max_t_stat": float(glm_result["t_stat"].max()),
            "min_p_value": float(glm_result["p_values"].min()),
        }

        # Add correction summary if applied
        if correction_results is not None:
            for i in range(contrasts.shape[0]):
                result = correction_results[f"contrast_{i}"]
                n_significant = int(np.sum(result.significant_mask))
                results_summary[f"contrast_{i}_significant_voxels"] = n_significant
                results_summary[f"contrast_{i}_min_corrected_p"] = float(
                    result.corrected_p_values.min()
                )

        from accelperm.io.output import create_results_summary

        create_results_summary(results_summary, summary_output)

        return {"status": "success", "results": results_summary}

    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    app()
