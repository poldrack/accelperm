"""Command-line interface for AccelPerm."""

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from accelperm.backends.cpu import CPUBackend
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
        "cpu",
        "--backend",
        "-b",
        help="Backend to use (cpu, mps, cuda)",
        case_sensitive=False,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """
    Run General Linear Model analysis with permutation testing.

    This command performs GLM analysis on neuroimaging data with the specified
    design matrix and contrasts, similar to FSL's randomise command.
    """
    # Validate backend
    valid_backends = ["cpu", "mps", "cuda"]
    if backend.lower() not in valid_backends:
        console.print(
            f"[red]Error: Invalid backend '{backend}'. Valid options: {', '.join(valid_backends)}[/red]"
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
    }

    if verbose:
        console.print("[green]Running GLM analysis...[/green]")
        console.print(f"Input: {input_file}")
        console.print(f"Design: {design_file}")
        console.print(f"Contrasts: {contrast_file}")
        console.print(f"Output: {output_dir}")
        console.print(f"Backend: {backend}")

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

        # Initialize backend
        if config["verbose"]:
            console.print(f"Initializing {config['backend'].upper()} backend...")

        if config["backend"] == "cpu":
            backend = CPUBackend()
        elif config["backend"] == "mps":
            from accelperm.backends.mps import MPSBackend

            backend = MPSBackend()
        else:
            # Only CPU and MPS backends are implemented
            raise ValueError(f"Backend '{config['backend']}' not yet implemented")

        if not backend.is_available():
            raise RuntimeError(f"Backend '{config['backend']}' is not available")

        # Run GLM
        if config["verbose"]:
            console.print("Computing GLM statistics...")

        glm_result = backend.compute_glm(Y, X, contrasts)

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

            # Save p-value map
            p_output = config["output_dir"] / f"pvals_{contrast_name}.nii.gz"
            output_writer.save_p_value_map(p_map, affine, p_output)

        # Create summary
        summary_output = config["output_dir"] / "results_summary.txt"
        results_summary = {
            "n_voxels": Y.shape[0],
            "n_subjects": Y.shape[1],
            "n_regressors": X.shape[1],
            "n_contrasts": contrasts.shape[0],
            "backend": config["backend"],
            "max_t_stat": float(glm_result["t_stat"].max()),
            "min_p_value": float(glm_result["p_values"].min()),
        }

        from accelperm.io.output import create_results_summary

        create_results_summary(results_summary, summary_output)

        return {"status": "success", "results": results_summary}

    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    app()
