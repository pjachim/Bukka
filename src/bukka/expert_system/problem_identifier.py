from bukka.data_management.dataset import Dataset
from bukka.expert_system.problems import ProblemsToSolve, Problem
from bukka.expert_system import implemented_solutions as sol


class ProblemIdentifier:
    """Scan a `Dataset` and identify problems to solve.

    The identifier inspects dataset-level properties (multivariate
    issues) and per-feature properties (univariate issues) and
    collects `Problem` instances into `self.problems_to_solve`.
    """

    def __init__(self, dataset: Dataset, target_column: str | None, problem_type: str = "auto") -> None:
        """Create a ProblemIdentifier for `dataset`.

        Args:
            dataset: A `Dataset` instance that exposes a `backend`
                with inspection helper methods and `feature_columns`.
            target_column: The name of the target column in the dataset. if None, then clustering is assumed.
            problem_type: ML problem type specification ('auto', 'binary_classification', 
                'multiclass_classification', 'regression', 'clustering'). Defaults to 'auto'.
        """
        self.dataset: Dataset = dataset
        self.target_column: str | None = target_column
        self.problem_type: str = problem_type
        self.problems_to_solve: ProblemsToSolve = ProblemsToSolve()

    def identify_problems(self) -> None:
        """Identify problems in the dataset and populate `problems_to_solve`."""
        self.multivariate_problems()
        self.univariate_problems()
        self.identify_ml_problem()

    def multivariate_problems(self) -> None:
        """Detect multivariate issues and add matching `Problem`s.

        Checks dataset-level diagnostics such as multicollinearity and
        strong correlations and appends `Problem` instances (with
        suggested solutions) to `self.problems_to_solve`.
        """
        if self.dataset.has_multicollinearity():
            # Check if multivariate_solutions module exists
            if hasattr(sol, 'multivariate_solutions') and hasattr(sol.multivariate_solutions, 'remove_multicollinear_features'):
                solutions = [sol.multivariate_solutions.remove_multicollinear_features]
            else:
                solutions = []
            
            problem = Problem(
                problem_name="Multicollinearity",
                description="The dataset contains multicollinear features.",
                solutions=solutions,
                features=[],
                problem_type="transformer"
            )
            self.problems_to_solve.add_problem(problem)

    def univariate_problems(self) -> None:
        """Inspect each feature for univariate problems.

        Iterates over `dataset.feature_columns` and delegates
        per-feature checks to `_identify_univariate_problems`.
        """
        for feature in self.dataset.feature_columns:
            self._identify_univariate_problems(feature)

    def _identify_univariate_problems(self, feature: str) -> None:
        """Identify and record univariate problems for `feature`.

        Args:
            feature: Name of the feature/column to inspect.
        """
        # Identify null/missing value problems
        if self.dataset.get_column_null_count(feature):
            problem = Problem(
                problem_name="Null Values",
                description=f"The feature '{feature}' contains null values.",
                features=[feature],
                solutions=[
                    sol.null_solutions.mean_solution,
                    sol.null_solutions.median_solution,
                ],
                problem_type="transformer",
            )
            self.problems_to_solve.add_problem(problem)

        # Numeric-type specific checks (outliers)
        if self.dataset.type_of_column(feature) in ["int", "float"]:
            if self.dataset.has_outliers(feature):
                problem = Problem(
                    problem_name="Outliers",
                    description=f"The feature '{feature}' contains outlier values.",
                    features=[feature],
                    solutions=[
                        sol.outlier_solutions.remove_outliers,
                        sol.outlier_solutions.cap_outliers,
                    ],
                    problem_type="transformer",
                )
                self.problems_to_solve.add_problem(problem)

        # String/categorical-type specific checks
        if self.dataset.type_of_column(feature) == "string":
            if self.dataset.has_inconsistent_categorical_data(feature):
                problem = Problem(
                    problem_name="Inconsistent Categorical Data",
                    description=f"The feature '{feature}' contains inconsistent categorical data.",
                    features=[feature],
                    solutions=[
                        sol.categorical_solutions.standardize_categories,
                        sol.categorical_solutions.encode_categories,
                    ],
                    problem_type="transformer",
                )
                self.problems_to_solve.add_problem(problem)

    def identify_ml_problem(self) -> str:
        """Identify the type of machine learning needed."""
        if self.target_column is None:
            self.ml_problem = Problem(
                problem_name="Clustering",
                description="Without a label, unsupervised clustering is needed.",
                features=[],
                solutions=[sol.clustering_solutions.clustering_analysis],
                problem_type="model",
            )
            return
        elif self.dataset.type_of_column(self.target_column) in ["int", "float"]:
            if self.dataset.get_unq_count(self.target_column) > 20:
                self.ml_problem = Problem(
                    problem_name="Regression",
                    description="The target variable is continuous.",
                    features=[self.target_column],
                    solutions=[sol.regression_solutions.regression_analysis],
                    problem_type="model",
                )
        
        # This means classification, as target is not None and not regression. Now let's see if it's binary or multi-class.
        if self.dataset.get_unq_count(self.target_column) == 2:
            self.ml_problem = Problem( 
                problem_name="Binary Classification",
                description="The target variable has two distinct classes.",
                features=[self.target_column],
                solutions=[sol.classification_solutions.binary_classification],
                problem_type="model",
            )
        else:
            self.ml_problem = Problem(
                problem_name="Multi-class Classification",
                description="The target variable has more than two distinct classes.",
                features=[self.target_column],
                solutions=[sol.classification_solutions.multi_class_classification],
                problem_type="model",
            )