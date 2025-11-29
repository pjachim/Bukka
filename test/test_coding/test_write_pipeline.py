import importlib
from types import SimpleNamespace
from pathlib import Path
import tempfile

import pytest

from bukka.coding.write_pipeline import PipelineWriter
from bukka.expert_system.problems import Problem
from bukka.expert_system.solution import Solution


class TestPipelineWriter:
    def test_pipeline_generation_and_import_extraction(self, tmp_path):
        """PipelineWriter should accept pipeline steps, extract imports and build a ColumnTransformer/Pipeline.

        This test creates pipeline steps directly (solution, problem tuples) and
        verifies that the writer correctly generates imports and pipeline code.
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

        # Create pipeline steps directly
        pipeline_steps = [
            (transformer_sol, transformer_problem),
            (model_sol, model_problem)
        ]
        
        output_path = tmp_path / "pipeline.py"
        writer = PipelineWriter(pipeline_steps=pipeline_steps, output_path=output_path)
        
        # Verify imports are extracted correctly
        assert "from sklearn.preprocessing import StandardScaler" in writer.imports
        assert "from sklearn.linear_model import LogisticRegression" in writer.imports
        assert "from sklearn.compose import ColumnTransformer" in writer.imports
        assert "from sklearn.pipeline import Pipeline" in writer.imports

        # Verify instantiations are created
        assert len(writer.instantiations) == 2
        assert "mytransformer" in writer.instantiations
        assert "mymodel" in writer.instantiations

        # pipeline_definition should contain ColumnTransformer and Pipeline
        assert "ColumnTransformer(" in writer.pipeline_definition
        assert "pipeline = Pipeline(" in writer.pipeline_definition
        
        # Write the file
        writer.write()
        
        # Verify the file was created
        assert output_path.exists()
        
        # Verify the file contents
        content = output_path.read_text()
        assert "from sklearn.preprocessing import StandardScaler" in content
        assert "from sklearn.linear_model import LogisticRegression" in content
        assert "ColumnTransformer(" in content
        assert "pipeline = Pipeline(" in content

    def test_unique_variable_names(self, tmp_path):
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
        
        # Create pipeline steps directly
        pipeline_steps = [
            (sol1, problem1),
            (sol2, problem2)
        ]
        
        output_path = tmp_path / "pipeline.py"
        writer = PipelineWriter(pipeline_steps=pipeline_steps, output_path=output_path)
        
        # Check that instantiations have unique keys
        assert len(writer.instantiations) == 2
        keys = list(writer.instantiations.keys())
        assert keys[0] != keys[1]
        assert "imputer" in keys[0]
        assert "imputer" in keys[1]

