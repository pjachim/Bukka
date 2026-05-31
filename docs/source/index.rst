.. Bukka documentation master file

Welcome to Bukka's Documentation!
===================================

**Bukka** is a Django-inspired Python CLI tool that dramatically reduces the boilerplate 
and setup time for new Machine Learning (ML) projects. Just like Django's ``startproject`` 
command, Bukka lets you instantly scaffold a robust, standardized, and ready-to-use project 
infrastructure.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   usage_examples
   api_reference
   configuration

Features
--------

**Django-Inspired Structure**: Creates a logical, maintainable folder hierarchy optimized for ML workflows.

**Automated Environment Setup**: Automatically generates a Python virtual environment.

**Dependency Management**: Creates a starting requirements.txt file with essential ML packages.

**YAML Configuration**: Use configuration files for complex project setups.

**Intelligent Pipeline Generation**: Automatically generates ML pipelines based on dataset analysis.

Quick Start
-----------

Install Bukka via pip:

.. code-block:: bash

   pip install bukka

Create a new ML project:

.. code-block:: bash

   python -m bukka run --name titanic --dataset titanic.csv --target Survived

This command will:

1. Create the project folder
2. Set up a virtual environment
3. Generate initial dependency files
4. Install packages
5. Copy and split your dataset
6. Provide starter notebooks
7. Provide code imports from `titanic`, i.e. from `titanic.utils import ...` to make your code more modular and reusable.
8. Write data loading scripts

This command can be further enhanced by adding additional arguments, like:
 - `--dummy` or `--tpot`to generate a dummy pipeline or a TPOT pipeline.
 - `--config` to specify a YAML configuration file for more complex setups.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
