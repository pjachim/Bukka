"""Integration tests for the Bukka CLI.

These tests exercise the command-line entrypoint and the programmatic
`main()` function. They patch the environment builder and dataset split
behaviour to keep tests hermetic and fast. Generated pipeline files are
validated for syntax and are executed under a lightweight `sklearn` stub so
their top-level code runs without requiring external packages.
"""

from pathlib import Path
import subprocess
import sys
import textwrap
import runpy
import types

import pytest


class TestCLIIntegration:
    """Class-based pytest suite for CLI integration tests.

    Notes
    -----
    - Tests run the bukka entrypoint in a subprocess using `-c` to inject
      patches before invoking the application. This avoids network calls and
      real virtualenv creation.
    - After the CLI run the tests assert the dataset was copied and at
      least one generated pipeline file was written. Each pipeline file is
      compiled to verify syntax and then executed with a lightweight
      `sklearn` stub to ensure top-level code is runnable.
    """

    @staticmethod
    def _base_runner_code(csv_path: str, proj_path: str) -> str:
        """Return a Python -c script that patches environment and dataset.

        The returned code will:
        - make `EnvironmentBuilder.build_environment` a no-op
        - monkeypatch `Dataset.__init__` so the polars backend reads the CSV
          and writes train/test parquet files
        - disable expensive `ProblemIdentifier` checks
        - call `bukka.__main__.main(...)` to run the project creation

        Parameters
        ----------
        csv_path : str
            Path to the CSV dataset to use.
        proj_path : str
            Destination project path.
        """

        template = textwrap.dedent(
            f"""
            from bukka.logistics.environment.environment import EnvironmentBuilder
            EnvironmentBuilder.build_environment = lambda self: None

            # Patch Dataset.__init__ so the Polars backend reads the CSV and
            # writes minimal parquet train/test files so the rest of the
            # pipeline generation can proceed.
            from bukka.data_management import dataset as ds_module
            from bukka.data_management.wrapper.polars import PolarsOperations
            import polars as pl
            import pyarrow.parquet as pq

            def _patched_init(self, target_column, file_manager, dataframe_backend='polars', strata=None, stratify=True, train_size=0.8, feature_columns=None):
                self.file_manager = file_manager
                self.target_column = target_column
                self.backend = PolarsOperations()
                try:
                    self.backend.full_df = pl.read_csv(str(self.file_manager.dataset_path))
                except Exception:
                    pass
                train_file = self.file_manager.train_data / 'train.parquet'
                test_file = self.file_manager.test_data / 'test.parquet'
                # Use backend splitting which will write parquet files
                self.backend.split_dataset(train_path=train_file, test_path=test_file, target_column=target_column, strata=strata, train_size=train_size, stratify=stratify)
                # Build a minimal train_df proxy exposing get_column_names()
                class _DFProxy:
                    def __init__(self, cols):
                        self._cols = cols
                    def get_column_names(self):
                        return list(self._cols)

                try:
                    cols = [f.name for f in pq.read_schema(train_file)]
                except Exception:
                    cols = []
                self.backend.train_df = _DFProxy(cols)
                if feature_columns is None:
                    if target_column in cols:
                        self.feature_columns = [c for c in cols if c != target_column]
                    else:
                        self.feature_columns = cols
                else:
                    self.feature_columns = feature_columns
                try:
                    schema = pq.read_schema(train_file)
                    self.data_schema = {field.name: field.type for field in schema}
                except Exception:
                    self.data_schema = {}

            ds_module.Dataset.__init__ = _patched_init

            # Disable heavy/problematic expert system checks in tests
            from bukka.expert_system import problem_identifier as pid_mod
            pid_mod.ProblemIdentifier.multivariate_problems = lambda self: None
            pid_mod.ProblemIdentifier.univariate_problems = lambda self: None
            pid_mod.ProblemIdentifier._identify_ml_problem = lambda self: None

            from bukka.__main__ import main
            main(name={repr(str(proj_path))}, dataset={repr(str(csv_path))}, target='target')
            """
        )

        return template

    def _execute_pipeline_with_stub(self, pipeline_path: Path) -> None:
        """Execute the generated pipeline under a lightweight sklearn stub.

        The stub provides minimal classes for commonly used estimators and a
        `Pipeline` class so the pipeline top-level code can run without
        external dependencies.
        """

        stub_code = textwrap.dedent(
            """
            import sys, types, runpy

            # Minimal sklearn stub modules and classes
            sklearn = types.ModuleType('sklearn')
            sys.modules['sklearn'] = sklearn
            pipeline_mod = types.ModuleType('sklearn.pipeline')
            class Pipeline:
                def __init__(self, steps):
                    self.steps = steps
            pipeline_mod.Pipeline = Pipeline
            sys.modules['sklearn.pipeline'] = pipeline_mod

            tree = types.ModuleType('sklearn.tree')
            class DecisionTreeClassifier:
                def __init__(self, *a, **k):
                    pass
            tree.DecisionTreeClassifier = DecisionTreeClassifier
            sys.modules['sklearn.tree'] = tree

            ensemble = types.ModuleType('sklearn.ensemble')
            class RandomForestClassifier:
                def __init__(self, *a, **k):
                    pass
            ensemble.RandomForestClassifier = RandomForestClassifier
            sys.modules['sklearn.ensemble'] = ensemble

            linear = types.ModuleType('sklearn.linear_model')
            class LogisticRegression:
                def __init__(self, *a, **k):
                    pass
            linear.LogisticRegression = LogisticRegression
            sys.modules['sklearn.linear_model'] = linear

            # Execute the pipeline file
            runpy.run_path({repr(str(pipeline_path))}, run_name='__main__')
            """
        )

        proc = subprocess.run([sys.executable, "-c", stub_code], capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(f"Executing pipeline failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    @pytest.mark.parametrize("classes", [[0, 1, 0, 1], [0, 1, 2, 0]])
    def test_cli_creates_and_executes_pipeline_via_main(self, tmp_path: Path, classes: list[int]) -> None:
        """Invoke `bukka.__main__.main()` in a subprocess and validate output.

        This test calls the programmatic `main()` after applying the same
        patches as the in-process CLI. It asserts that dataset copying and
        pipeline generation occur and that the produced pipeline is
        syntactically valid and runnable under the sklearn stub.
        """

        csv = tmp_path / "sample.csv"
        lines = ["feature,target"]
        for i, c in enumerate(classes, start=1):
            lines.append(f"{i},{c}")
        csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

        proj = tmp_path / "proj"

        code = self._base_runner_code(str(csv), str(proj))

        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, f"CLI run failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

        # Validate files
        copied = proj / "data" / csv.name
        assert copied.exists(), f"Dataset not copied to project: {copied}"

        gen_dir = proj / "pipelines" / "generated"
        assert gen_dir.exists(), "Generated pipelines directory missing"

        pipeline_files = sorted(gen_dir.glob("pipeline_*.py"))
        assert pipeline_files, "No pipeline files generated"

        # Verify each pipeline is syntactically valid and runnable under stub
        for p in pipeline_files:
            # Syntax check
            src = p.read_text(encoding="utf-8")
            compile(src, str(p), 'exec')
            # Execute with stubbed sklearn
            self._execute_pipeline_with_stub(p)

    def test_cli_invocation_via_run_module(self, tmp_path: Path) -> None:
        """Simulate `python -m bukka` by running the module as `__main__`.

        This ensures the argument parsing branch and module execution path are
        exercised.
        """

        csv = tmp_path / "sample2.csv"
        csv.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")
        proj = tmp_path / "proj2"

        run_module_code = textwrap.dedent(
            f"""
            import runpy, sys
            # Apply same patches as before
            from bukka.logistics.environment.environment import EnvironmentBuilder
            EnvironmentBuilder.build_environment = lambda self: None
            from bukka.data_management import dataset as ds_module
            from bukka.data_management.wrapper.polars import PolarsOperations
            import polars as pl
            import pyarrow.parquet as pq

            def _patched_init(self, target_column, file_manager, dataframe_backend='polars', strata=None, stratify=True, train_size=0.8, feature_columns=None):
                self.file_manager = file_manager
                self.target_column = target_column
                self.backend = PolarsOperations()
                try:
                    self.backend.full_df = pl.read_csv(str(self.file_manager.dataset_path))
                except Exception:
                    pass
                train_file = self.file_manager.train_data / 'train.parquet'
                test_file = self.file_manager.test_data / 'test.parquet'
                self.backend.split_dataset(train_path=train_file, test_path=test_file, target_column=target_column, strata=strata, train_size=train_size, stratify=stratify)
                class _DFProxy:
                    def __init__(self, cols):
                        self._cols = cols
                    def get_column_names(self):
                        return list(self._cols)
                try:
                    cols = [f.name for f in pq.read_schema(train_file)]
                except Exception:
                    cols = []
                self.backend.train_df = _DFProxy(cols)
                if feature_columns is None:
                    if target_column in cols:
                        self.feature_columns = [c for c in cols if c != target_column]
                    else:
                        self.feature_columns = cols
                else:
                    self.feature_columns = feature_columns
                try:
                    schema = pq.read_schema(train_file)
                    self.data_schema = {field.name: field.type for field in schema}
                except Exception:
                    self.data_schema = {}

            ds_module.Dataset.__init__ = _patched_init
            from bukka.expert_system import problem_identifier as pid_mod
            pid_mod.ProblemIdentifier.multivariate_problems = lambda self: None
            pid_mod.ProblemIdentifier.univariate_problems = lambda self: None
            pid_mod.ProblemIdentifier._identify_ml_problem = lambda self: None

            # Simulate CLI args and run module as __main__
            sys.argv = ['-m', 'bukka', '--name', {repr(str(proj))}, '--dataset', {repr(str(csv))}, '--target', 'target']
            runpy.run_module('bukka', run_name='__main__')
            """
        )

        proc = subprocess.run([sys.executable, "-c", run_module_code], capture_output=True, text=True)
        assert proc.returncode == 0, f"Module-run CLI failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

        gen_dir = proj / "pipelines" / "generated"
        assert gen_dir.exists(), "Generated pipelines dir missing after module run"

        pipeline_files = sorted(gen_dir.glob("pipeline_*.py"))
        assert pipeline_files, "No pipeline files created by module run"

        for p in pipeline_files:
            compile(p.read_text(encoding='utf-8'), str(p), 'exec')
            self._execute_pipeline_with_stub(p)

