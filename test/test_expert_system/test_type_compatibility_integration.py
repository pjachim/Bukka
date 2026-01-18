"""Integration test demonstrating type compatibility in the full problem identification flow."""
import pytest
from bukka.expert_system.problems import Problem
from bukka.expert_system import implemented_solutions as sol


class TestTypeCompatibilityIntegration:
    """Integration tests for type compatibility across the expert system."""

    def test_numeric_column_only_gets_numeric_solutions(self):
        """Test that a numeric column problem only receives compatible numeric solutions.
        
        Examples
        --------
        >>> problem = Problem("Null Values", "Has nulls", ["age"])
        >>> problem.add_solution(sol.null_solutions.mean_solution, column_type="int")
        >>> problem.add_solution(sol.categorical_solutions.encode_categories, column_type="int")
        >>> assert len(problem.solutions) == 1  # Only mean_solution added
        """
        problem = Problem(
            problem_name="Null Values",
            description="Numeric column has null values",
            features=["age"]
        )
        
        # Try adding both numeric and categorical solutions
        problem.add_solution(sol.null_solutions.mean_solution, column_type="int")
        problem.add_solution(sol.null_solutions.median_solution, column_type="float")
        problem.add_solution(sol.categorical_solutions.encode_categories, column_type="int")
        problem.add_solution(sol.categorical_solutions.standardize_categories, column_type="float")
        
        # Should only have numeric solutions
        assert len(problem.solutions) == 2
        assert all(s.name in ["mean_imputer", "median_imputer"] for s in problem.solutions)

    def test_string_column_only_gets_string_solutions(self):
        """Test that a string column problem only receives compatible string solutions.
        
        Examples
        --------
        >>> problem = Problem("Categorical", "Has inconsistent data", ["category"])
        >>> problem.add_solution(sol.categorical_solutions.encode_categories, column_type="string")
        >>> problem.add_solution(sol.null_solutions.mean_solution, column_type="string")
        >>> assert len(problem.solutions) == 1  # Only encode_categories added
        """
        problem = Problem(
            problem_name="Inconsistent Categorical Data",
            description="String column has inconsistent values",
            features=["category"]
        )
        
        # Try adding both string and numeric solutions
        problem.add_solution(sol.categorical_solutions.encode_categories, column_type="string")
        problem.add_solution(sol.categorical_solutions.standardize_categories, column_type="string")
        problem.add_solution(sol.null_solutions.mean_solution, column_type="string")
        problem.add_solution(sol.null_solutions.median_solution, column_type="string")
        
        # Should only have string solutions
        assert len(problem.solutions) == 2
        assert all(s.name in ["encode_categories", "standardize_categories"] for s in problem.solutions)

    def test_outlier_problem_with_numeric_types(self):
        """Test that outlier solutions are only added for numeric types.
        
        Examples
        --------
        >>> problem = Problem("Outliers", "Has outliers", ["price"])
        >>> problem.add_solution(sol.outlier_solutions.cap_outliers, column_type="float")
        >>> problem.add_solution(sol.outlier_solutions.cap_outliers, column_type="string")
        >>> assert len(problem.solutions) == 1  # Only for float
        """
        problem = Problem(
            problem_name="Outliers",
            description="Column has outlier values",
            features=["price"]
        )
        
        # Add for numeric types
        problem.add_solution(sol.outlier_solutions.cap_outliers, column_type="int")
        problem.add_solution(sol.outlier_solutions.cap_outliers, column_type="float")
        
        # Try adding for string (should be rejected)
        problem.add_solution(sol.outlier_solutions.cap_outliers, column_type="string")
        
        # Should only be added for numeric types
        assert len(problem.solutions) == 2

    def test_mixed_type_dataset_gets_appropriate_solutions(self):
        """Test that problems for different column types get appropriate solutions.
        
        Examples
        --------
        >>> numeric_problem = Problem("Null", "Nulls in age", ["age"])
        >>> string_problem = Problem("Null", "Nulls in name", ["name"])
        >>> numeric_problem.add_solution(sol.null_solutions.mean_solution, "int")
        >>> string_problem.add_solution(sol.null_solutions.mean_solution, "string")
        >>> assert len(numeric_problem.solutions) == 1
        >>> assert len(string_problem.solutions) == 0
        """
        # Create problems for different column types
        age_problem = Problem(
            problem_name="Null Values in Age",
            description="Age column (int) has nulls",
            features=["age"]
        )
        
        price_problem = Problem(
            problem_name="Outliers in Price",
            description="Price column (float) has outliers",
            features=["price"]
        )
        
        category_problem = Problem(
            problem_name="Inconsistent Categories",
            description="Category column (string) has inconsistent values",
            features=["category"]
        )
        
        # Add solutions with appropriate types
        age_problem.add_solution(sol.null_solutions.mean_solution, column_type="int")
        age_problem.add_solution(sol.null_solutions.median_solution, column_type="int")
        
        price_problem.add_solution(sol.outlier_solutions.cap_outliers, column_type="float")
        
        category_problem.add_solution(sol.categorical_solutions.encode_categories, column_type="string")
        category_problem.add_solution(sol.categorical_solutions.standardize_categories, column_type="string")
        
        # Verify each problem has the right solutions
        assert len(age_problem.solutions) == 2
        assert all(s.name in ["mean_imputer", "median_imputer"] for s in age_problem.solutions)
        
        assert len(price_problem.solutions) == 1
        assert price_problem.solutions[0].name == "cap_outliers"
        
        assert len(category_problem.solutions) == 2
        assert all(s.name in ["encode_categories", "standardize_categories"] for s in category_problem.solutions)

    def test_text_solutions_only_for_string_columns(self):
        """Test that text vectorization solutions only work with string columns.
        
        Examples
        --------
        >>> problem = Problem("Text Processing", "Need to vectorize", ["description"])
        >>> problem.add_solution(sol.text_solutions.tfidf_solution, column_type="string")
        >>> problem.add_solution(sol.text_solutions.tfidf_solution, column_type="int")
        >>> assert len(problem.solutions) == 1  # Only for string
        """
        text_problem = Problem(
            problem_name="Text Vectorization",
            description="Description column needs vectorization",
            features=["description"]
        )
        
        # Add text solutions with string type
        text_problem.add_solution(sol.text_solutions.tfidf_solution, column_type="string")
        text_problem.add_solution(sol.text_solutions.countvectorizer_solution, column_type="string")
        
        # Try with numeric types (should be rejected)
        text_problem.add_solution(sol.text_solutions.tfidf_solution, column_type="int")
        text_problem.add_solution(sol.text_solutions.countvectorizer_solution, column_type="float")
        
        # Should only have the string-type additions
        assert len(text_problem.solutions) == 2
        assert all(s.name in ["tfidf_vectorizer", "count_vectorizer"] for s in text_problem.solutions)
