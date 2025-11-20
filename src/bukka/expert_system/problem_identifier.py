from bukka.data_management.dataset import Dataset
from bukka.expert_system.problems import ProblemsToSolve, Problem
from bukka.expert_system import implemented_solutions as sol

# Class that scans the dataset and identifies problems based on attributes in the dataset. 
class ProblemIdentifier:
    def __init__(self, dataset:Dataset):
        self.dataset = dataset
        self.problems_to_solve = ProblemsToSolve()

    def multivariate_problems(self):
        # Check for multivariate problems here (e.g., correlation issues, multicollinearity, etc.)
        if self.dataset.backend.has_multicollinearity():
            problem = Problem(
                problem_name="Multicollinearity",
                description="The dataset contains multicollinear features.",
                solutions=[sol.multivariate_solutions.remove_multicollinear_features]
            )
            self.problems_to_solve.add_problem(problem)

        if self.dataset.backend.has_strong_correlations():
            problem = Problem(
                problem_name="Strong Correlations",
                description="The dataset contains strongly correlated features.",
                solutions=[sol.multivariate_solutions.handle_strong_correlations]
            )
            self.problems_to_solve.add_problem(problem)

    def univariate_problems(self):
        for feature in self.dataset.feature_columns:
            self._identify_univariate_problems(feature)

    def _identify_univariate_problems(self, feature):
        # Identify if there are null_problems:
        if self.dataset.backend.get_column_null_count(feature):
            problem = Problem(
                problem_name="Null Values",
                description=f"The feature '{feature}' contains null values.",
                solutions=[sol.null_solutions.impute_missing_values, sol.null_solutions.remove_rows_with_nulls]
            )
            self.problems_to_solve.add_problem(problem)

        if self.dataset.backend.type_of_column(feature) in ['int', 'float']:
            # Identify outlier problems:
            if self.dataset.backend.has_outliers(feature):
                problem = Problem(
                    problem_name="Outliers",
                    description=f"The feature '{feature}' contains outlier values.",
                    solutions=[sol.outlier_solutions.remove_outliers, sol.outlier_solutions.cap_outliers]
                )
                self.problems_to_solve.add_problem(problem)

        if self.dataset.backend.type_of_column(feature) == 'string':
            # Identify inconsistent categorical data problems:
            if self.dataset.backend.has_inconsistent_categorical_data(feature):
                problem = Problem(
                    problem_name="Inconsistent Categorical Data",
                    description=f"The feature '{feature}' contains inconsistent categorical data.",
                    solutions=[sol.categorical_solutions.standardize_categories, sol.categorical_solutions.encode_categories]
                )
                self.problems_to_solve.add_problem(problem)