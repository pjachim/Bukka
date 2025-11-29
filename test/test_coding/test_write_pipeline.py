"""Unit tests for PipelineWriter after refactor to use TemplateBaseClass.

This test suite validates the refactored PipelineWriter which now:
- Inherits from TemplateBaseClass for template-based code generation
- Uses write_code() method instead of write()
- Separates concerns into _fetch_imports(), _parse_pipeline_steps(), _build_*() methods
- Distinguishes between transformers (column-specific) and manipulators (multi-column)

Known bugs documented in tests (DO NOT FIX HERE):
1. FULL_TEMPLATE has hardcoded imports (Pipeline, ColumnTransformer) that aren't
   exposed in writer.imports, making the imports set incomplete. See
   test_pipeline_generation_and_import_extraction for details.
2. Multiple transformers on the same columns create parallel ColumnTransformer entries
   rather than a chained Pipeline, which may not apply transforms sequentially.
   See test_multiple_transformers_same_columns for details.
"""
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
        
        # Verify imports are extracted correctly from solutions
        assert "from sklearn.preprocessing import StandardScaler" in writer.imports
        assert "from sklearn.linear_model import LogisticRegression" in writer.imports
        # NOTE: BUG - ColumnTransformer and Pipeline imports are hardcoded in FULL_TEMPLATE
        # but not exposed in writer.imports. This makes writer.imports incomplete.
        # These should either be added to writer.imports or extracted from the template.

        # Verify instantiations are created
        assert len(writer.instantiations) == 2
        assert "mytransformer" in writer.instantiations
        assert "mymodel" in writer.instantiations

        # pipeline_definition should contain ColumnTransformer and Pipeline
        assert "ColumnTransformer(" in writer.pipeline_definition
        assert "pipeline = Pipeline(" in writer.pipeline_definition
        
        # Write the file using the parent class method
        writer.write_code()
        
        # Verify the file was created
        assert output_path.exists()
        
        # Verify the file contents
        content = output_path.read_text()
        assert "from sklearn.preprocessing import StandardScaler" in content
        assert "from sklearn.linear_model import LogisticRegression" in content
        assert "from sklearn.pipeline import Pipeline" in content
        assert "from sklearn.compose import ColumnTransformer" in content
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

    def test_manipulator_problem_type(self, tmp_path):
        """Test that manipulator problem types are handled correctly."""
        
        # Create a manipulator solution (e.g., PCA for dimensionality reduction)
        manipulator_sol = Solution(
            name="pca",
            function_import="from sklearn.decomposition import PCA",
            function_name="PCA",
            function_kwargs={"n_components": 5},
        )
        
        manipulator_problem = Problem(
            problem_name="Dimensionality Reduction",
            description="Reduce dimensions across all features",
            features=["feature1", "feature2", "feature3"],
            solutions=[manipulator_sol],
            problem_type="manipulator",
        )
        
        model_sol = Solution(
            name="classifier",
            function_import="from sklearn.ensemble import RandomForestClassifier",
            function_name="RandomForestClassifier",
            function_kwargs={},
        )
        
        model_problem = Problem(
            problem_name="Classification",
            description="Classification task",
            features=["target"],
            solutions=[model_sol],
            problem_type="model",
        )
        
        pipeline_steps = [
            (manipulator_sol, manipulator_problem),
            (model_sol, model_problem)
        ]
        
        output_path = tmp_path / "pipeline.py"
        writer = PipelineWriter(pipeline_steps=pipeline_steps, output_path=output_path)
        
        # Verify that manipulator is in the manipulators list
        assert len(writer.manipulators) == 1
        assert "pca" in writer.manipulators
        
        # Verify that transformers list is empty (no column-specific transformers)
        assert len(writer.transformers) == 0
        
        # Write and verify content
        writer.write_code()
        content = output_path.read_text()
        
        # Manipulators should NOT be in ColumnTransformer
        assert "('pca', pca)" in content
        assert "from sklearn.decomposition import PCA" in content

    def test_mixed_transformer_and_manipulator(self, tmp_path):
        """Test pipeline with both transformers and manipulators."""
        
        # Transformer for specific columns
        scaler_sol = Solution(
            name="scaler",
            function_import="from sklearn.preprocessing import StandardScaler",
            function_name="StandardScaler",
            function_kwargs={},
        )
        
        scaler_problem = Problem(
            problem_name="Scaling",
            description="Scale numeric features",
            features=["age", "income"],
            solutions=[scaler_sol],
            problem_type="transformer",
        )
        
        # Manipulator for all features
        pca_sol = Solution(
            name="pca",
            function_import="from sklearn.decomposition import PCA",
            function_name="PCA",
            function_kwargs={"n_components": 10},
        )
        
        pca_problem = Problem(
            problem_name="Dimensionality",
            description="Reduce dimensions",
            features=[],
            solutions=[pca_sol],
            problem_type="manipulator",
        )
        
        # Model
        model_sol = Solution(
            name="model",
            function_import="from sklearn.linear_model import LogisticRegression",
            function_name="LogisticRegression",
            function_kwargs={},
        )
        
        model_problem = Problem(
            problem_name="Classification",
            description="Binary classification",
            features=["target"],
            solutions=[model_sol],
            problem_type="model",
        )
        
        pipeline_steps = [
            (scaler_sol, scaler_problem),
            (pca_sol, pca_problem),
            (model_sol, model_problem)
        ]
        
        output_path = tmp_path / "pipeline.py"
        writer = PipelineWriter(pipeline_steps=pipeline_steps, output_path=output_path)
        
        # Verify categorization
        assert len(writer.transformers) == 1
        assert len(writer.manipulators) == 1
        assert writer.model_step == "model"
        
        # Write and verify structure
        writer.write_code()
        content = output_path.read_text()
        
        # Should have ColumnTransformer with scaler
        assert "preprocessor = ColumnTransformer([" in content
        assert "('scaler', scaler, ['age', 'income'])" in content
        
        # Pipeline should include preprocessor, pca, and model
        assert "('preprocessor', preprocessor)" in content
        assert "('pca', pca)" in content
        assert "('model', model)" in content

    def test_empty_pipeline_steps(self, tmp_path):
        """Test that empty pipeline steps are handled gracefully."""
        
        output_path = tmp_path / "pipeline.py"
        writer = PipelineWriter(pipeline_steps=[], output_path=output_path)
        
        # Should have empty collections
        assert len(writer.imports) == 0
        assert len(writer.instantiations) == 0
        assert len(writer.transformers) == 0
        assert len(writer.manipulators) == 0
        assert writer.model_step is None
        
        # Should still be able to write (though it will be an empty pipeline)
        writer.write_code()
        assert output_path.exists()

    def test_problem_type_default_behavior(self, tmp_path):
        """Test that steps without problem_type are treated as manipulators."""
        
        # Solution without explicit problem_type
        sol = Solution(
            name="custom_step",
            function_import="from custom_module import CustomTransformer",
            function_name="CustomTransformer",
            function_kwargs={},
        )
        
        # Problem without problem_type attribute
        problem = Problem(
            problem_name="Custom Processing",
            description="Custom processing step",
            features=["feature1"],
            solutions=[sol],
            problem_type=None,  # Explicitly None
        )
        
        pipeline_steps = [(sol, problem)]
        
        output_path = tmp_path / "pipeline.py"
        writer = PipelineWriter(pipeline_steps=pipeline_steps, output_path=output_path)
        
        # Should be treated as manipulator (default behavior)
        assert len(writer.manipulators) == 1
        assert "custom_step" in writer.manipulators
        assert len(writer.transformers) == 0

    def test_multiple_transformers_same_columns(self, tmp_path):
        """Test multiple transformers can target the same columns."""
        
        # First transformer
        imputer_sol = Solution(
            name="imputer",
            function_import="from sklearn.impute import SimpleImputer",
            function_name="SimpleImputer",
            function_kwargs={"strategy": "mean"},
        )
        
        imputer_problem = Problem(
            problem_name="Handle Nulls",
            description="Impute missing values",
            features=["age", "income"],
            solutions=[imputer_sol],
            problem_type="transformer",
        )
        
        # Second transformer on same columns
        scaler_sol = Solution(
            name="scaler",
            function_import="from sklearn.preprocessing import StandardScaler",
            function_name="StandardScaler",
            function_kwargs={},
        )
        
        scaler_problem = Problem(
            problem_name="Scale Features",
            description="Scale numeric features",
            features=["age", "income"],
            solutions=[scaler_sol],
            problem_type="transformer",
        )
        
        pipeline_steps = [
            (imputer_sol, imputer_problem),
            (scaler_sol, scaler_problem)
        ]
        
        output_path = tmp_path / "pipeline.py"
        writer = PipelineWriter(pipeline_steps=pipeline_steps, output_path=output_path)
        
        # Both should be in transformers
        assert len(writer.transformers) == 2
        
        # NOTE: POTENTIAL BUG - Having multiple transformers on the same columns
        # will create a ColumnTransformer with both, but the second transformer
        # won't receive the output of the first. This might not be the intended behavior.
        # Consider if transformers on same columns should be chained in a Pipeline
        # inside the ColumnTransformer instead.
        
        writer.write_code()
        content = output_path.read_text()
        
        # Both should appear in the ColumnTransformer
        assert "('imputer', imputer, ['age', 'income'])" in content
        assert "('scaler', scaler, ['age', 'income'])" in content

    def test_instantiation_formatting(self, tmp_path):
        """Test that instantiations are formatted correctly."""
        
        sol = Solution(
            name="MyModel",
            function_import="from sklearn.ensemble import RandomForestClassifier",
            function_name="RandomForestClassifier",
            function_kwargs={"n_estimators": 100, "random_state": 42, "max_depth": 5},
        )
        
        problem = Problem(
            problem_name="Classification",
            description="Multi-class classification",
            features=["target"],
            solutions=[sol],
            problem_type="model",
        )
        
        pipeline_steps = [(sol, problem)]
        
        output_path = tmp_path / "pipeline.py"
        writer = PipelineWriter(pipeline_steps=pipeline_steps, output_path=output_path)
        
        writer.write_code()
        content = output_path.read_text()
        
        # Verify the instantiation includes all kwargs
        assert "mymodel = RandomForestClassifier(" in content
        # The exact formatting depends on Solution.fetch_instantiation()
        # Just verify the variable name and function are present

