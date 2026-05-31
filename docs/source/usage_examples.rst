Usage Examples
==============

This page stays close to the CLI. If you are using Bukka day to day, this is the page to copy from.

Basic Project Creation
----------------------

Create a project with a dataset:

.. code-block:: bash

   python -m bukka run --name iris_classifier --dataset iris.csv --target species

Create the same project through a config file when the command gets longer:

.. code-block:: bash

   python -m bukka init-config
   python -m bukka run --config bukka_config.yaml

Common project shapes
---------------------

Binary classification:

.. code-block:: bash

   python -m bukka run -n fraud_detection -d transactions.csv -t is_fraud \
       --backend pandas --problem-type binary_classification

Regression:

.. code-block:: bash

   python -m bukka run -n housing_prices -d housing.csv -t price \
       --problem-type regression --train-size 0.75

Clustering:

.. code-block:: bash

   python -m bukka run -n customer_segments -d customers.csv \
       --problem-type clustering --backend polars

Tuning the scaffold
-------------------

The CLI also supports a few switches for when the defaults are not enough:

.. code-block:: bash

   python -m bukka run -n my_project -d data.csv --skip-venv
   python -m bukka run -n my_project -d data.csv --strata gender age_group
   python -m bukka run -n my_project -d data.csv --mlflow
   python -m bukka run -n my_project -d data.csv --dummy
   python -m bukka run -n my_project -d data.csv --tpot

The programmatic API exists, but it is the long way around. Most people should stay on the CLI.

Next steps
----------

* Read :doc:`configuration` for YAML-based setup.
* Read :doc:`api_reference` only if you are extending Bukka itself.
