Installation
============

Requirements
------------

AccelPerm requires Python 3.12 or later.

**Core Requirements:**

* numpy >= 1.24.0
* scipy >= 1.11.0
* nibabel >= 5.0.0
* torch >= 2.0.0 (for GPU acceleration)
* pydantic >= 2.0.0
* typer >= 0.9.0
* rich >= 13.0.0

**GPU Requirements:**

For CUDA support:
* CUDA Toolkit 11.7+
* cuDNN 8.0+
* NVIDIA driver 515+
* PyTorch 2.0+ with CUDA support

For Apple Silicon (MPS) support:
* macOS 12.3+ (Monterey or later)
* PyTorch 2.0+ with MPS support
* Xcode Command Line Tools

Standard Installation
--------------------

Install from PyPI::

    pip install accelperm

This will install AccelPerm with all required dependencies.

Development Installation
-----------------------

For development, clone the repository and install in editable mode::

    git clone https://github.com/poldrack/accelperm.git
    cd accelperm

    # Using uv (recommended)
    uv sync --all-extras

    # Or using pip
    pip install -e ".[dev]"

This installs AccelPerm in development mode with all optional dependencies
including testing, documentation, and profiling tools.

Verifying Installation
---------------------

Test your installation::

    accelperm --version

    # Check GPU availability
    python -c "import torch; print('CUDA:', torch.cuda.is_available())"
    python -c "import torch; print('MPS:', torch.backends.mps.is_available())"

Troubleshooting
--------------

**GPU not detected:**

1. Verify GPU drivers are properly installed
2. Check PyTorch installation with GPU support::

    python -c "import torch; print(torch.__version__)"

3. For CUDA issues, verify CUDA toolkit installation::

    nvcc --version

4. For MPS issues on macOS, ensure you have macOS 12.3+ and the latest PyTorch

**Memory issues:**

AccelPerm automatically manages memory and will fall back to CPU computation
if GPU memory is insufficient. For large datasets, consider:

1. Reducing batch size in configuration
2. Using chunked processing
3. Increasing virtual memory/swap space

**Import errors:**

If you encounter import errors, ensure all dependencies are properly installed::

    pip install --upgrade pip
    pip install -e ".[dev]"
