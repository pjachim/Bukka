import importlib
from types import SimpleNamespace

import pytest

from bukka.coding.write_pipeline import PipelineWriter
from bukka.expert_system.problems import Problem
from bukka.expert_system.solution import Solution


class TestPipelineWriter:
    def test_pipeline_generation_and_import_extraction(self):
        """PipelineWriter should select processors, extract imports and build a ColumnTransformer/Pipeline.

        This test builds a minimal fake `ProblemIdentifier` with one transformer
        solution and one model solution. The behavior under test
        is deterministic because each problem's solution list contains a
        single element.
        """

        # A transformer solution with proper Solution interface
        transformer_sol = Solution(
            name="MyTransformer",
            function_import="from sklearn.preprocessing import StandardScaler",
            function_name="StandardScaler",
            function_kwargs={},
        )
        
        # Problem with transformer type
        transformer_problem = Problem(
            problem_name="Scaling",
            description="Feature needs scaling",
            features=["feature1"],
            solutions=[transformer_sol],
            problem_type="transformer",
        )

        # A model solution
        model_sol = Solution(
            name="MyModel",
            function_import="from sklearn.linear_model import LogisticRegression",
            function_name="LogisticRegression",
            function_kwargs={"max_iter": 1000},
        )
        
        # Problem with model type
        model_problem = Problem(
            problem_name="Binary Classification",
            description="Binary classification task",
            features=["target"],
            solutions=[model_sol],
            problem_type="model",
        )

        problems_to_solve = SimpleNamespace(problems=[transformer_problem])
        ml_problem = model_problem
        pid = SimpleNamespace(problems_to_solve=problems_to_solve, ml_problem=ml_problem)

        writer = PipelineWriter(pid)
        steps, imports = writer.write()

        # Steps should be a list of tuples (solution, problem)
        assert isinstance(steps, list)
        assert len(steps) == 2
        assert all(isinstance(step, tuple) and len(step) == 2 for step in steps)

        # Imports should include solution imports plus ColumnTransformer and Pipeline
        assert "from sklearn.preprocessing import StandardScaler" in imports
        assert "from sklearn.linear_model import LogisticRegression" in imports
        assert "from sklearn.compose import ColumnTransformer" in imports
        assert "from sklearn.pipeline import Pipeline" in imports

        # pipeline_definition should contain ColumnTransformer and Pipeline
        assert "ColumnTransformer(" in writer.pipeline_definition
        assert "pipeline = Pipeline(" in writer.pipeline_definition

    def test_unique_variable_names(self):
        """Test that duplicate solution names get unique variable names."""
        
        # Two solutions with the same name
        sol1 = Solution(
            name="imputer",
            function_import="from sklearn.impute import SimpleImputer",
            function_name="SimpleImputer",
            function_kwargs={"strategy": "mean"},
        )
        
        sol2 = Solution(
            name="imputer",
            function_import="from sklearn.impute import SimpleImputer",
            function_name="SimpleImputer",
            function_kwargs={"strategy": "median"},
        )
        
        problem1 = Problem(
            problem_name="Null Values 1",
            description="Feature 1 has nulls",
            features=["feature1"],
            solutions=[sol1],
            problem_type="transformer",
        )
        
        problem2 = Problem(
            problem_name="Null Values 2",
            description="Feature 2 has nulls",
            features=["feature2"],
            solutions=[sol2],
            problem_type="transformer",
        )
        
        problems_to_solve = SimpleNamespace(problems=[problem1, problem2])
        ml_problem = SimpleNamespace(solutions=[])
        pid = SimpleNamespace(problems_to_solve=problems_to_solve, ml_problem=ml_problem)
        
        writer = PipelineWriter(pid)
        steps, imports = writer.write()
        
        # Check that instantiations have unique keys
        assert len(writer.instantiations) == 2
        keys = list(writer.instantiations.keys())
        assert keys[0] != keys[1]
        assert "imputer" in keys[0]
        assert "imputer" in keys[1]

