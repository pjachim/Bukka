from bukka.expert_system import solution

# Binary Classification Solutions that only work for two-class problems
logistic_regression_solution = solution.Solution(
    name="logistic_regression_binary",
    explanation="A logistic regression model specifically for binary classification tasks.",
    function_kwargs={
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 100,
    },
    function_import="from sklearn.linear_model import LogisticRegression",
    function_name="LogisticRegression",
)

