from bukka.expert_system import solution

remove_outliers_solution = solution.Solution(
    name="remove_outliers",
    explanation="Detects and removes outlier values using Isolation Forest algorithm.",
    function_kwargs={
        "contamination": 0.1,
        "random_state": 42
    },
    function_import="from sklearn.ensemble import IsolationForest",
    function_name="IsolationForest",
)

cap_outliers_solution = solution.Solution(
    name="cap_outliers",
    explanation="Applies robust scaling to handle outliers using median and IQR statistics.",
    function_kwargs={
        "quantile_range": (25.0, 75.0)
    },
    function_import="from sklearn.preprocessing import RobustScaler",
    function_name="RobustScaler",
)

# For ProblemIdentifier compatibility
remove_outliers = remove_outliers_solution
cap_outliers = cap_outliers_solution
