from bukka.expert_system import solution

remove_outliers_solution = solution.Solution(
    name="remove_outliers",
    explanation="Removes outlier values from a numerical feature using the IQR method.",
    function_kwargs={
        "method": "IQR"
    },
    function_import="from bukka.preprocessing.outliers import remove_outliers",
    function_name="remove_outliers",
)

cap_outliers_solution = solution.Solution(
    name="cap_outliers",
    explanation="Caps outlier values in a numerical feature at the IQR boundaries.",
    function_kwargs={
        "method": "IQR"
    },
    function_import="from bukka.preprocessing.outliers import cap_outliers",
    function_name="cap_outliers",
)

# For ProblemIdentifier compatibility
remove_outliers = remove_outliers_solution
cap_outliers = cap_outliers_solution
