from bukka.expert_system import solution

cap_outliers_solution = solution.Solution(
    name="cap_outliers",
    explanation="Applies robust scaling to handle outliers using median and IQR statistics.",
    function_kwargs={
        "quantile_range": (25.0, 75.0)
    },
    function_import="from sklearn.preprocessing import RobustScaler",
    function_name="RobustScaler",
    compatible_types=["int", "float"],
)

# For ProblemIdentifier compatibility
cap_outliers = cap_outliers_solution
