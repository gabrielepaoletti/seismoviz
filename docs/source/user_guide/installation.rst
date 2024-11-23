Installation
============

This document provides detailed instructions for installing the SeismoViz package. The installation process has been streamlined to ensure compatibility and ease of integration into various scientific computing environments.

Prerequisites
-------------
Before proceeding with the installation, ensure that you have a compatible Python environment. SeismoViz requires Python >= 3.11. 

It's strongly recommended creating a virtual environment to maintain a clean workspace and avoid dependency conflicts. Tools such as ``conda`` or ``venv`` are suitable for creating isolated Python environments.

.. tabs::
    .. tab:: pip
        .. code-block:: bash
            
            python -m venv seismoviz
    
    .. tab:: conda
        .. code-block:: bash
            
            conda create --name seismoviz

Stable release
--------------
The SeismoViz package is conveniently available for installation via ``pip``, the Python package manager. To install the latest stable release from the Python Package Index (PyPI), execute the following command in your terminal or command prompt:

.. code-block:: bash

    pip install seismoviz

This command fetches and installs the most recent stable version of SeismoViz, along with its required dependencies.

.. warning::
    At present, installation via ``pip`` is unavailable, as the project is still in its early stages. To install SeismoViz, please follow the instructions for the *Development version* installation.

Development version
-------------------
For those interested in accessing the latest features and developments, the cutting-edge version of SeismoViz can be installed directly from the source code. Begin by cloning the project repository from GitHub with:

.. code-block:: bash

    git clone https://github.com/gabrielepaoletti/seismoviz.git

Once the repository is cloned, navigate to the project's root directory:

.. code-block:: bash

    cd seismoviz

Complete the installation by running:

.. code-block:: bash

    pip install .

This sequence of commands installs the current development version of SeismoViz from the cloned repository into your Python environment.

Should you encounter any issues during the installation process, feel free to submit an issue on the GitHub repository.