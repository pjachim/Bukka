from bukka.expert_system import solution

null_solution = solution.Solution(
    name="null_imputer",
    explanation="Imputes missing values with a constant value (e.g., zero or 'missing').",
    function_kwargs={
        "strategy": "constant",
        "fill_value": 0
    },
    function_import="from sklearn.impute import SimpleImputer",
    function_name="SimpleImputer",
)

mean_solution = solution.Solution(
    name="mean_imputer",
    explanation="Imputes missing numerical values with the mean of the respective feature.",
    function_kwargs={
        "strategy": "mean"
    },
    function_import="from sklearn.impute import SimpleImputer",
    function_name="SimpleImputer",
)

median_solution = solution.Solution(
    name="median_imputer",
    explanation="Imputes missing numerical values with the median of the respective feature.",
    function_kwargs={
        "strategy": "median"
    },
    function_import="from sklearn.impute import SimpleImputer",
    function_name="SimpleImputer",
)