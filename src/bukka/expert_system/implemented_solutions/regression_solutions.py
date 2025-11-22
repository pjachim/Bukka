from bukka.expert_system import solution

# Regression models
linear_regression_solution = solution.Solution(
    name="linear_regression",
    explanation="A linear regression model that assumes a linear relationship between input features and the target variable.",
    function_kwargs={
        "fit_intercept": True,
        "normalize": False,
    },
    function_import="from sklearn.linear_model import LinearRegression",
    function_name="LinearRegression",
)

ridge_regression_solution = solution.Solution(
    name="ridge_regression",
    explanation="A linear regression model with L2 regularization to prevent overfitting.",
    function_kwargs={
        "alpha": 1.0,
        "fit_intercept": True,
        "normalize": False,
    },
    function_import="from sklearn.linear_model import Ridge",
    function_name="Ridge",
)