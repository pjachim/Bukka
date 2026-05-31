Configuration
=============

Bukka supports YAML configuration when the command line starts to get unwieldy.

Configuration File Format
--------------------------

Bukka uses YAML configuration files for project setup. Generate a template with:

.. code-block:: bash

   python -m bukka init-config

Default Configuration
^^^^^^^^^^^^^^^^^^^^^

The default template looks like this:

.. code-block:: yaml

   # Bukka Project Configuration Template
   
   # Project settings
   project:
     name: my_ml_project
     dataset: data/train.csv
     target: target_column
     skip_venv: false
     enable_mlflow: false
     mlflow_tracking_uri: null
   
   # Data processing settings
   data:
     backend: polars
     train_size: 0.8
     stratify: true
     strata: null
   
   # Problem specification
   problem:
     type: auto  # or: binary_classification, multiclass_classification, regression, clustering

Configuration Sections
----------------------

Project section
^^^^^^^^^^^^^^^

The ``project`` section holds the basic project fields:

.. list-table::
   :header-rows: 1
   :widths: 20 15 50 15

   * - Parameter
     - Type
     - Description
     - Default
   * - ``name``
     - string
     - Project name and directory.
     - Required
   * - ``dataset``
     - string
     - Path to the dataset file.
     - Optional
   * - ``target``
     - string
     - Target column name.
     - Optional
   * - ``skip_venv``
     - boolean
     - Skip virtual environment creation.
     - ``false``
   * - ``enable_mlflow``
     - boolean
     - Enable MLflow tracking.
     - ``false``
   * - ``mlflow_tracking_uri``
     - string
     - Custom MLflow tracking URI.
     - ``null``

Example:

.. code-block:: yaml

   project:
     name: fraud_detection
     dataset: transactions.csv
     target: is_fraud
     skip_venv: false
     enable_mlflow: true

Data section
^^^^^^^^^^^^

The ``data`` section controls dataset handling:

.. list-table::
   :header-rows: 1
   :widths: 20 15 50 15

   * - Parameter
     - Type
     - Description
     - Default
   * - ``backend``
     - string
     - Dataframe backend.
     - ``polars``
   * - ``train_size``
     - float
     - Train/test split ratio.
     - ``0.8``
   * - ``stratify``
     - boolean
     - Enable stratified sampling.
     - ``true``
   * - ``strata``
     - list
     - Columns for stratification.
     - ``null``

Example:

.. code-block:: yaml

   data:
     backend: pandas
     train_size: 0.75
     stratify: true
     strata: [gender, age_group]

Problem section
^^^^^^^^^^^^^^^

The ``problem`` section tells Bukka how to treat the project:

.. list-table::
   :header-rows: 1
   :widths: 20 15 50 15

   * - Parameter
     - Type
     - Description
     - Default
   * - ``type``
     - string
     - ML problem type.
     - ``auto``

Valid problem types:

* ``auto`` - Automatic detection
* ``binary_classification`` - Binary classification
* ``multiclass_classification`` - Multi-class classification
* ``regression`` - Regression
* ``clustering`` - Clustering

Example:

.. code-block:: yaml

   problem:
     type: binary_classification

Command-line arguments
----------------------

Every configuration option can also be passed on the command line.

Priority order
^^^^^^^^^^^^^^

When both a config file and CLI arguments are provided:

1. **Command-line arguments** (highest priority)
2. **Configuration file**
3. **Default values** (lowest priority)

Example:

.. code-block:: bash

   # Config file sets backend to 'polars', CLI overrides to 'pandas'
   python -m bukka run --config config.yaml --backend pandas

Global options
^^^^^^^^^^^^^^

.. code-block:: bash

   python -m bukka [OPTIONS] COMMAND [ARGS]

Options:

* ``--help`` - Show help message

Init-config command
^^^^^^^^^^^^^^^^^^^

Generate a configuration template:

.. code-block:: bash

   python -m bukka init-config [--output FILE]

Options:

* ``-o, --output FILE`` - Output path (default: ``bukka_config.yaml``)

Run command
^^^^^^^^^^^

Create a Bukka project:

.. code-block:: bash

   python -m bukka run [OPTIONS]

**Configuration:**

* ``-c, --config FILE`` - Path to YAML configuration file

**Project settings:**

* ``-n, --name NAME`` - Project name (required unless using --config)
* ``-d, --dataset FILE`` - Dataset file path
* ``-t, --target COLUMN`` - Target column name
* ``-sv, --skip-venv`` - Skip virtual environment creation
* ``--mlflow`` - Enable MLflow tracking
* ``--mlflow-tracking-uri URI`` - MLflow tracking URI

**Data processing:**

* ``-b, --backend NAME`` - Dataframe backend (default: polars)
* ``--train-size RATIO`` - Train/test split ratio (default: 0.8)
* ``--stratify`` - Enable stratified split (default: True)
* ``--no-stratify`` - Disable stratified split
* ``--strata COLUMNS`` - Stratification columns

**Problem specification:**

* ``-p, --problem-type TYPE`` - ML problem type (default: auto)

Configuration examples
----------------------

Example 1: Basic Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   project:
     name: iris_classifier
     dataset: iris.csv
     target: species

The rest of the old examples have been left out on purpose. The CLI pages are the primary reference for day-to-day use, and the API details are in :doc:`api_reference`.
   
   data:
     backend: polars
     train_size: 0.8
   
   problem:
     type: multiclass_classification

Usage:

.. code-block:: bash

   python -m bukka run --config iris_config.yaml

Example 2: Regression with Custom Split
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   project:
     name: house_prices
     dataset: housing.csv
     target: price
   
   data:
     backend: pandas
     train_size: 0.7
     stratify: false
   
   problem:
     type: regression

Usage:

.. code-block:: bash

   python -m bukka run --config housing_config.yaml

Example 3: Clustering with No Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   project:
     name: customer_segments
     dataset: customers.csv
     skip_venv: true
   
   data:
     backend: polars
   
   problem:
     type: clustering

Usage:

.. code-block:: bash

   python -m bukka run --config clustering_config.yaml

Example 4: Stratified Sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   project:
     name: medical_prediction
     dataset: patient_data.csv
     target: diagnosis
   
   data:
     backend: polars
     train_size: 0.8
     stratify: true
     strata: [gender, age_group, severity]
   
   problem:
     type: binary_classification

Usage:

.. code-block:: bash

   python -m bukka run --config medical_config.yaml

Example 5: MLflow Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   project:
     name: tracked_experiment
     dataset: experiment_data.csv
     target: outcome
     enable_mlflow: true
     mlflow_tracking_uri: "http://localhost:5000"
   
   data:
     backend: polars
     train_size: 0.8
   
   problem:
     type: binary_classification

Usage:

.. code-block:: bash

   python -m bukka run --config mlflow_config.yaml

Configuration Validation
-------------------------

Bukka validates all configuration values:

Project Name Validation
^^^^^^^^^^^^^^^^^^^^^^^

* Cannot be empty
* Cannot contain invalid characters: ``< > : " | ? *``

.. code-block:: bash

   # ❌ Invalid
   python -m bukka run --name "my:project"
   # Error: Project name contains invalid characters

Dataset Path Validation
^^^^^^^^^^^^^^^^^^^^^^^

* File must exist
* Path must point to a file (not a directory)

.. code-block:: bash

   # ❌ Invalid
   python -m bukka run --name proj --dataset nonexistent.csv
   # Error: Dataset file not found

Backend Validation
^^^^^^^^^^^^^^^^^^

* Must be a supported backend

Supported backends (via Narwhals):

* ``polars`` (default, recommended)
* ``pandas``
* ``pyarrow``
* ``modin``
* ``cudf``
* ``dask``

.. code-block:: bash

   # ❌ Invalid
   python -m bukka run --name proj --backend invalid
   # Error: Backend 'invalid' not supported

Problem Type Validation
^^^^^^^^^^^^^^^^^^^^^^^

* Must be a recognized ML problem type

Valid types:

* ``auto``
* ``binary_classification``
* ``multiclass_classification``
* ``regression``
* ``clustering``

Train Size Validation
^^^^^^^^^^^^^^^^^^^^^

* Must be between 0.0 and 1.0

.. code-block:: bash

   # ❌ Invalid
   python -m bukka run --name proj --train-size 1.5
   # Error: train_size must be between 0 and 1

Environment Variables
---------------------

Bukka respects standard Python environment variables:

* ``PYTHONPATH`` - Additional module search paths
* ``VIRTUAL_ENV`` - Current virtual environment path

Best Practices
--------------

1. **Version Control**: Commit your ``bukka_config.yaml`` files to version control.

2. **Template Reuse**: Create template configs for common project types.

3. **Documentation**: Add comments to your YAML files to document choices.

4. **Validation**: Test your config with ``init-config`` before running.

5. **Overrides**: Use CLI arguments for temporary overrides, config files for permanent settings.

6. **Modularity**: Create separate configs for different experiments.

Troubleshooting
---------------

Config File Not Found
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Error: Configuration file not found: config.yaml

Solution: Check the file path is correct and the file exists.

Invalid YAML Syntax
^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Error: Invalid YAML syntax in config file

Solution: Validate your YAML with a linter or:

.. code-block:: bash

   python -c "import yaml; yaml.safe_load(open('config.yaml'))"

Missing Required Fields
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Error: Project name is required

Solution: Either provide ``--name`` or set ``project.name`` in the config file.

Next Steps
----------

* See :doc:`usage_examples` for practical examples
* Check :doc:`getting_started` for installation
* Explore :doc:`api_reference` for programmatic usage
