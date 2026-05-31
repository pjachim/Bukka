.. Bukka documentation master file

Bukka documentation
===================

Bukka is a CLI for creating a machine learning project scaffold from a dataset or a config file.
It handles the repetitive setup work so you can start with code that is already organized.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   usage_examples
   configuration
   api_reference

Quick start
-----------

Install Bukka:

.. code-block:: bash

   pip install bukka

Create a new project:

.. code-block:: bash

   python -m bukka run --name titanic --dataset titanic.csv --target Survived

What the command does:

1. Creates the project folder.
2. Sets up a virtual environment unless you pass ``--skip-venv``.
3. Writes the initial configuration and dependency files.
4. Copies the dataset into the new project.
5. Splits the dataset into train and test sets.
6. Writes starter files such as the data reader and notebook.
7. Adds optional helpers when you pass flags like ``--mlflow``, ``--dummy``, or ``--tpot``.

For the command reference and the config file format, see the other pages in this guide.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
