Getting Started
===============

Installation
------------

Bukka is available on PyPI and can be installed with pip:

.. code-block:: bash

   pip install bukka

Requirements:

* Python >= 3.10
* PyYAML >= 6.0
* narwhals >= 1.0.0
* numpy >= 2.0.0

Creating Your First Project
---------------------------

The quickest way to start is with the ``run`` command:

.. code-block:: bash

   python -m bukka run --name my_project --dataset data.csv --target target_column

This creates the standard project scaffold:

* A project folder with the Bukka layout.
* A virtual environment unless you pass ``--skip-venv``.
* A copied dataset and train/test splits.
* Starter files for configuration, data loading, and notebooks.
* Optional extras such as MLflow, dummy, or TPOT files.

Example: Binary Classification
------------------------------

Predicting Titanic survival looks like this:

.. testcode::

   import subprocess
   import tempfile
   import os
   
   # Create a sample CSV file
   with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
       f.write('PassengerId,Survived,Pclass,Name,Sex,Age\\n')
       f.write('1,0,3,Mr. Owen Harris,male,22\\n')
       f.write('2,1,1,Mrs. John Bradley,female,38\\n')
       f.write('3,1,3,Miss. Laina,female,26\\n')
       csv_file = f.name
   
   # The command would be:
   # python -m bukka run --name titanic --dataset titanic.csv --target Survived

.. testcleanup::

   import os
   if 'csv_file' in locals():
       os.unlink(csv_file)

Example: Regression Project
----------------------------

Predicting a numeric target such as house price:

.. testcode::

   import tempfile
   
   # Create a sample housing dataset
   with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
       f.write('sqft,bedrooms,bathrooms,price\\n')
       f.write('1500,3,2,300000\\n')
       f.write('2000,4,3,450000\\n')
       f.write('1200,2,1,250000\\n')
       csv_file = f.name
   
   # The command would be:
   # python -m bukka run --name housing --dataset housing.csv --target price --problem-type regression

.. testcleanup::

   import os
   if 'csv_file' in locals():
       os.unlink(csv_file)

Using the Project Class
-----------------------

You can also use Bukka programmatically if you need to script the setup step:

.. testcode::

   from bukka.project import Project
   
   # Create a project instance
   proj = Project(
       name="my_ml_project",
       dataset_path=None,  # Can be added later
       backend="polars",
       problem_type="auto"
   )
   
   # Verify project properties
   assert proj.name == "my_ml_project"
   assert proj.backend == "polars"
   assert proj.problem_type == "auto"
   print("Project created successfully")

.. testoutput::

   Project created successfully

Project Structure
-----------------

When you create a project, Bukka generates this structure:

.. code-block:: text

   my_project/
   ├── .venv/                      # Virtual environment
   ├── data/                       # Data storage
   │   ├── <dataset>               # Original dataset
   │   ├── test/                   # Test split
   │   └── train/                  # Training split
   ├── pipelines/                  # ML Pipelines
   │   ├── __init__.py
   │   ├── baseline/               # Baseline models
   │   ├── candidate/              # Custom pipelines
   │   └── generated/              # Auto-generated pipelines
   ├── notebooks/                  # Jupyter notebooks
   │   └── starter_notebook.ipynb
   ├── utils/                      # Utilities
   ├── pyproject.toml
   └── requirements.txt

Next Steps
----------

* Read :doc:`usage_examples` for CLI patterns.
* Read :doc:`configuration` if you prefer YAML over flags.
* Read :doc:`api_reference` only if you need the modules behind the CLI.
