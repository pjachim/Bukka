import pytest

from bukka.expert_system.problem_identifier import ProblemIdentifier
from bukka.expert_system.problems import ProblemsToSolve
from types import SimpleNamespace

# Ensure implemented_solutions has the attributes expected by ProblemIdentifier
import bukka.expert_system.implemented_solutions as impl

# inject minimal placeholders for modules/attributes that may not be exported
if not hasattr(impl, "multivariate_solutions"):
    impl.multivariate_solutions = SimpleNamespace(
        remove_multicollinear_features=lambda *a, **k: None,
        handle_strong_correlations=lambda *a, **k: None,
    )
if not hasattr(impl, "clustering_solutions"):
    impl.clustering_solutions = SimpleNamespace(clustering_analysis=lambda *a, **k: None)
if not hasattr(impl, "classification_solutions"):
    impl.classification_solutions = SimpleNamespace(
        binary_classification=lambda *a, **k: None,
        multi_class_classification=lambda *a, **k: None,
    )
if not hasattr(impl, "regression_solutions"):
    impl.regression_solutions = SimpleNamespace(regression_analysis=lambda *a, **k: None)


class DummyBackend:
    def __init__(self, **kwargs):
        self._vals = kwargs

    def has_multicollinearity(self):
        return self._vals.get("has_multicollinearity", False)

    def has_strong_correlations(self):
        return self._vals.get("has_strong_correlations", False)

    def get_column_null_count(self, col):
        return self._vals.get("nulls", {}).get(col, 0)

    def type_of_column(self, col):
        types = self._vals.get("types", {})
        return types.get(col, "string")

    def has_outliers(self, col):
        return self._vals.get("outliers", {}).get(col, False)

    def has_inconsistent_categorical_data(self, col):
        return self._vals.get("inconsistent", {}).get(col, False)

    def get_unq_count(self, col):
        return self._vals.get("unique_counts", {}).get(col, 0)


class DummyDataset:
    def __init__(self, backend, feature_columns):
        self.backend = backend
        self.feature_columns = feature_columns


def test_multivariate_detection():
    backend = DummyBackend(has_multicollinearity=True, has_strong_correlations=True)
    ds = DummyDataset(backend=backend, feature_columns=[])

    pi = ProblemIdentifier(ds, target_column=None)
    pi.multivariate_problems()

    names = [p.problem_name for p in pi.problems_to_solve.problems]
    assert "Multicollinearity" in names
    assert "Strong Correlations" in names


def test_univariate_detection_nulls_outliers_categorical():
    backend = DummyBackend(
        nulls={"f1": 3},
        types={"f1": "int", "f2": "float", "f3": "string"},
        outliers={"f2": True},
        inconsistent={"f3": True},
    )
    ds = DummyDataset(backend=backend, feature_columns=["f1", "f2", "f3"])
    pi = ProblemIdentifier(ds, target_column=None)
    pi.univariate_problems()

    names = [p.problem_name for p in pi.problems_to_solve.problems]
    assert "Null Values" in names
    assert "Outliers" in names
    assert "Inconsistent Categorical Data" in names


def test_ml_problem_identification_clustering_and_classification():
    # Clustering when target is None
    backend = DummyBackend()
    ds = DummyDataset(backend=backend, feature_columns=[])
    pi = ProblemIdentifier(ds, target_column=None)
    pi._identify_ml_problem()
    assert pi.ml_problem.problem_name == "Clustering"

    # Binary classification when unique count == 2
    backend2 = DummyBackend(types={"t": "int"}, unique_counts={"t": 2}, unique_counts_ignored=True)
    # Note: ProblemIdentifier uses get_unq_count; pass via unique_counts
    backend2 = DummyBackend(types={"t": "int"}, unique_counts={"t": 2})
    ds2 = DummyDataset(backend=backend2, feature_columns=[])
    pi2 = ProblemIdentifier(ds2, target_column="t")
    pi2._identify_ml_problem()
    assert pi2.ml_problem.problem_name == "Binary Classification"

    # Multi-class expected when unique count > 2 (current implementation sets Multi-class)
    backend3 = DummyBackend(types={"t": "int"}, unique_counts={"t": 50})
    ds3 = DummyDataset(backend=backend3, feature_columns=[])
    pi3 = ProblemIdentifier(ds3, target_column="t")
    pi3._identify_ml_problem()
    assert pi3.ml_problem.problem_name == "Multi-class Classification"
