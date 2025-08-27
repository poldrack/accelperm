AccelPerm Documentation
======================

Welcome to AccelPerm's documentation!

AccelPerm is a GPU-accelerated permutation testing toolkit for neuroimaging data,
inspired by FSL's randomise but leveraging modern GPU architectures (CUDA and Apple
Metal Performance Shaders) for 10-100x performance improvements while maintaining
<0.001% deviation from the reference implementation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   contributing

Features
--------

* **High Performance**: GPU acceleration with CUDA and MPS support
* **Statistical Accuracy**: <0.001% deviation from FSL randomise
* **Easy to Use**: Drop-in replacement with familiar CLI
* **Cross-Platform**: Support for Linux, macOS, and Windows
* **Extensible**: Clean architecture for custom analyses

Installation
------------

AccelPerm can be installed using pip::

    pip install accelperm

For development installation with all dependencies::

    git clone https://github.com/poldrack/accelperm.git
    cd accelperm
    uv sync --all-extras

Quick Start
-----------

Basic one-sample t-test::

    accelperm -i input.nii -o output -1 -T -c 3.0 -v 1000

With exchangeability blocks::

    accelperm -i input.nii -o output -1 -T -c 3.0 -v 1000 -e design.grp

Multiple regression with TFCE::

    accelperm -i input.nii -o output -d design.mat -t design.con -T -v 1000

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
