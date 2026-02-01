Usage Examples
==============

This page provides comprehensive examples of using Bukka for various ML tasks.

Basic Project Creation
----------------------

Minimal Project
^^^^^^^^^^^^^^^

Create a project structure without a dataset:

.. testcode::

   from bukka.project import Project
   
   # Create minimal project
   proj = Project(name="minimal_project")
   
   assert proj.name == "minimal_project"
   assert proj.dataset_path is None
   assert proj.backend == "polars"  # Default backend
   print("Minimal project initialized")

.. testoutput::

   Minimal project initialized

Project with Dataset
^^^^^^^^^^^^^^^^^^^^

Create a project with a dataset path and target column:

.. testcode::

   from bukka.project import Project
   
   proj = Project(
       name="iris_classifier",
       dataset_path="/path/to/iris.csv",
       target_column="species"
   )
   
   assert proj.name == "iris_classifier"
   assert proj.target_column == "species"
   print("Project with dataset configured")

.. testoutput::

   Project with dataset configured

Advanced Configuration
----------------------

Custom Backend Selection
^^^^^^^^^^^^^^^^^^^^^^^^

Choose a specific dataframe backend:

.. testcode::

   from bukka.project import Project
   
   # Use pandas backend
   proj = Project(
       name="pandas_project",
       backend="pandas"
   )
   
   assert proj.backend == "pandas"
   print(f"Using backend: {proj.backend}")

.. testoutput::

   Using backend: pandas

Custom Train/Test Split
^^^^^^^^^^^^^^^^^^^^^^^^

Specify a custom train/test split ratio:

.. testcode::

   from bukka.project import Project
   
   proj = Project(
       name="custom_split",
       train_size=0.7  # 70% training, 30% testing
   )
   
   assert proj.train_size == 0.7
   print(f"Train size: {proj.train_size}")

.. testoutput::

   Train size: 0.7

Problem Type Specification
^^^^^^^^^^^^^^^^^^^^^^^^^^

Explicitly specify the ML problem type:

.. testcode::

   from bukka.project import Project
   
   # Binary classification
   binary_proj = Project(
       name="binary_clf",
       problem_type="binary_classification"
   )
   
   # Regression
   regression_proj = Project(
       name="regression_proj",
       problem_type="regression"
   )
   
   # Clustering
   clustering_proj = Project(
       name="clustering_proj",
       problem_type="clustering"
   )
   
   assert binary_proj.problem_type == "binary_classification"
   assert regression_proj.problem_type == "regression"
   assert clustering_proj.problem_type == "clustering"
   print("Problem types configured successfully")

.. testoutput::

   Problem types configured successfully

Stratified Sampling
^^^^^^^^^^^^^^^^^^^

Configure stratified train/test splitting:

.. testcode::

   from bukka.project import Project
   
   # Enable stratification
   proj = Project(
       name="stratified_project",
       stratify=True,
       strata=["gender", "age_group"]
   )
   
   assert proj.stratify is True
   assert proj.strata == ["gender", "age_group"]
   print("Stratification configured")

.. testoutput::

   Stratification configured

Skip Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^

Skip virtual environment creation for faster setup:

.. testcode::

   from bukka.project import Project
   
   proj = Project(
       name="no_venv_project",
       skip_venv=True
   )
   
   assert proj.skip_venv is True
   print("Virtual environment will be skipped")

.. testoutput::

   Virtual environment will be skipped

CLI Examples
------------

Quick Start
^^^^^^^^^^^

.. code-block:: bash

   # Basic project
   python -m bukka run --name my_project --dataset data.csv --target price

Classification Project
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Binary classification with pandas
   python -m bukka run --name fraud_detection \
       --dataset transactions.csv \
       --target is_fraud \
       --backend pandas \
       --problem-type binary_classification

   # Multiclass classification
   python -m bukka run --name digit_classifier \
       --dataset digits.csv \
       --target label \
       --problem-type multiclass_classification

Regression Project
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # House price prediction
   python -m bukka run --name housing_prices \
       --dataset housing.csv \
       --target price \
       --problem-type regression \
       --train-size 0.75

Clustering Project
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Customer segmentation (no target column)
   python -m bukka run --name customer_segments \
       --dataset customers.csv \
       --problem-type clustering \
       --backend polars

Configuration File Usage
------------------------

Generate Configuration Template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create default config template
   python -m bukka init-config

   # Create with custom name
   python -m bukka init-config --output my_config.yaml

Use Configuration File
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run with config file
   python -m bukka run --config bukka_config.yaml

   # Override specific config values
   python -m bukka run --config bukka_config.yaml \
       --backend pandas \
       --train-size 0.9

Working with Datasets
---------------------

Dataset Class Usage
^^^^^^^^^^^^^^^^^^^

The Dataset class is used internally by Bukka to manage data loading and splitting.
Here's an example of how it's typically used:

.. code-block:: python

   from bukka.data_management.dataset import Dataset
   from bukka.utils.files.file_manager import FileManager
   from pathlib import Path
   
   # Create FileManager instance
   fm = FileManager(
       project_path=Path("/path/to/project"),
       orig_dataset=Path("data.csv")
   )
   
   # Load and split dataset
   ds = Dataset(
       target_column='target',
       file_manager=fm,
       backend='polars'
   )
   
   # Access train and test splits
   print(f"Training data shape: {ds.train_df.shape}")
   print(f"Test data shape: {ds.test_df.shape}")

The Dataset class automatically handles:

* Loading data from various formats (CSV, Parquet, etc.)
* Splitting into train/test sets
* Stratified sampling when specified
* Schema extraction and validation

Complete Example Workflow
--------------------------

Here's a complete example combining multiple features:

.. code-block:: bash

   # Step 1: Generate config
   python -m bukka init-config --output titanic_config.yaml

   # Step 2: Edit the config file (titanic_config.yaml):
   # project:
   #   name: titanic_survival
   #   dataset: titanic.csv
   #   target: Survived
   # data:
   #   backend: polars
   #   train_size: 0.8
   #   stratify: true
   # problem:
   #   type: binary_classification

   # Step 3: Create the project
   python -m bukka run --config titanic_config.yaml

   # Step 4: Navigate to project and activate environment
   # cd titanic_survival
   # source .venv/bin/activate  # On Linux/Mac
   # .venv\\Scripts\\activate    # On Windows

   # Step 5: Start working with Jupyter
   # jupyter notebook notebooks/starter_notebook.ipynb

Best Practices
--------------

1. **Use Configuration Files**: For projects you'll recreate or share, use YAML configs.

2. **Version Control**: Add your ``bukka_config.yaml`` to version control, but not ``.venv/`` or ``data/``.

3. **Consistent Naming**: Use descriptive project names that reflect the task (e.g., ``fraud_detection`` not ``project1``).

4. **Backend Selection**: 
   - Use ``polars`` for fast data processing (default)
   - Use ``pandas`` for compatibility with existing code
   - Other backends coming soon!

5. **Problem Type**: Let Bukka auto-detect when unsure, or specify explicitly for better pipeline generation.

6. **Data Splits**: Keep default 80/20 split unless you have a specific reason to change it.

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**Project already exists**

.. code-block:: text

   Error: Directory 'my_project' already exists

Solution: Use a different name or remove the existing directory.

**Dataset not found**

.. code-block:: text

   Error: Dataset file not found: data.csv

Solution: Provide the full or relative path to your dataset file.

**Invalid backend**

.. code-block:: text

   Error: Backend 'invalid' not supported

Solution: Use a supported backend (currently: ``polars``).

Next Steps
----------

* Explore the :doc:`api_reference` for detailed documentation
* Check out :doc:`configuration` for YAML config options
* Read :doc:`getting_started` for installation instructions
