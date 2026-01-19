"""Test type compatibility logic for solutions."""
import pytest
from bukka.expert_system.solution import Solution
from bukka.expert_system.problems import Problem


class TestSolutionTypeCompatibility:
    """Test suite for Solution type compatibility features."""

    def test_solution_default_compatible_types(self):
        """Test that solutions default to all types when compatible_types is not specified.
        
        Examples
        --------
        >>> solution = Solution("test", "import test", "test", {})
        >>> assert solution.compatible_types == ["int", "float", "string"]
        """
        solution = Solution(
            name="test_solution",
            function_import="from sklearn.impute import SimpleImputer",
            function_name="SimpleImputer",
            function_kwargs={"strategy": "mean"}
        )
        assert solution.compatible_types == ["int", "float", "string"]

    def test_solution_explicit_compatible_types(self):
        """Test that solutions can specify explicit compatible types.
        
        Examples
        --------
        >>> solution = Solution("test", "import test", "test", {}, compatible_types=["int"])
        >>> assert solution.compatible_types == ["int"]
        """
        solution = Solution(
            name="numeric_solution",
            function_import="from sklearn.impute import SimpleImputer",
            function_name="SimpleImputer",
            function_kwargs={"strategy": "mean"},
            compatible_types=["int", "float"]
        )
        assert solution.compatible_types == ["int", "float"]


class TestProblemAddSolutionTypeChecking:
    """Test suite for Problem.add_solution type checking."""

    def test_add_solution_without_type_checking(self):
        """Test that solutions can be added without type checking when column_type is None.
        
        Examples
        --------
        >>> problem = Problem("Test", "A test problem", ["col1"])
        >>> solution = Solution("test", "import test", "test", {}, compatible_types=["int"])
        >>> problem.add_solution(solution)
        >>> assert len(problem.solutions) == 1
        """
        problem = Problem(
            problem_name="Test Problem",
            description="A test problem",
            features=["test_feature"]
        )
        solution = Solution(
            name="test_solution",
            function_import="from sklearn.impute import SimpleImputer",
            function_name="SimpleImputer",
            function_kwargs={"strategy": "mean"},
            compatible_types=["int", "float"]
        )
        
        # Add without type checking
        problem.add_solution(solution)
        assert len(problem.solutions) == 1

    def test_add_solution_with_compatible_type(self):
        """Test that compatible solutions are added when column type matches.
        
        Examples
        --------
        >>> problem = Problem("Test", "A test problem", ["col1"])
        >>> solution = Solution("test", "import test", "test", {}, compatible_types=["int", "float"])
        >>> problem.add_solution(solution, column_type="int")
        >>> assert len(problem.solutions) == 1
        """
        problem = Problem(
            problem_name="Null Values",
            description="Feature contains null values",
            features=["numeric_feature"]
        )
        solution = Solution(
            name="mean_imputer",
            function_import="from sklearn.impute import SimpleImputer",
            function_name="SimpleImputer",
            function_kwargs={"strategy": "mean"},
            compatible_types=["int", "float"]
        )
        
        # Add with compatible type
        problem.add_solution(solution, column_type="int")
        assert len(problem.solutions) == 1

    def test_add_solution_with_incompatible_type(self):
        """Test that incompatible solutions are not added when column type doesn't match.
        
        Examples
        --------
        >>> problem = Problem("Test", "A test problem", ["col1"])
        >>> solution = Solution("test", "import test", "test", {}, compatible_types=["int", "float"])
        >>> problem.add_solution(solution, column_type="string")
        >>> assert len(problem.solutions) == 0
        """
        problem = Problem(
            problem_name="Null Values",
            description="Feature contains null values",
            features=["string_feature"]
        )
        solution = Solution(
            name="mean_imputer",
            function_import="from sklearn.impute import SimpleImputer",
            function_name="SimpleImputer",
            function_kwargs={"strategy": "mean"},
            compatible_types=["int", "float"]
        )
        
        # Try to add with incompatible type
        problem.add_solution(solution, column_type="string")
        assert len(problem.solutions) == 0

    def test_add_multiple_solutions_with_type_filtering(self):
        """Test that only compatible solutions are added from a list.
        
        Examples
        --------
        >>> problem = Problem("Test", "A test problem", ["col1"])
        >>> numeric_sol = Solution("mean", "import", "Mean", {}, compatible_types=["int", "float"])
        >>> string_sol = Solution("encode", "import", "Encoder", {}, compatible_types=["string"])
        >>> problem.add_solution(numeric_sol, column_type="int")
        >>> problem.add_solution(string_sol, column_type="int")
        >>> assert len(problem.solutions) == 1
        """
        problem = Problem(
            problem_name="Test Problem",
            description="A test problem",
            features=["int_feature"]
        )
        
        numeric_solution = Solution(
            name="mean_imputer",
            function_import="from sklearn.impute import SimpleImputer",
            function_name="SimpleImputer",
            function_kwargs={"strategy": "mean"},
            compatible_types=["int", "float"]
        )
        
        string_solution = Solution(
            name="category_encoder",
            function_import="from sklearn.preprocessing import OrdinalEncoder",
            function_name="OrdinalEncoder",
            function_kwargs={},
            compatible_types=["string"]
        )
        
        # Add both solutions with int column type
        problem.add_solution(numeric_solution, column_type="int")
        problem.add_solution(string_solution, column_type="int")
        
        # Only the numeric solution should be added
        assert len(problem.solutions) == 1
        assert problem.solutions[0].name == "mean_imputer"

    def test_backward_compatibility_no_compatible_types(self):
        """Test that solutions without compatible_types attribute still work.
        
        Examples
        --------
        >>> problem = Problem("Test", "A test problem", ["col1"])
        >>> # Simulate old-style solution without compatible_types
        >>> class OldSolution:
        ...     def __init__(self):
        ...         self.name = "old"
        >>> old_sol = OldSolution()
        >>> problem.add_solution(old_sol, column_type="int")
        >>> assert len(problem.solutions) == 1
        """
        problem = Problem(
            problem_name="Test Problem",
            description="A test problem",
            features=["test_feature"]
        )
        
        # Create a minimal solution-like object without compatible_types
        class OldStyleSolution:
            def __init__(self):
                self.name = "old_solution"
        
        old_solution = OldStyleSolution()
        
        # Should add without type checking
        problem.add_solution(old_solution, column_type="int")
        assert len(problem.solutions) == 1
