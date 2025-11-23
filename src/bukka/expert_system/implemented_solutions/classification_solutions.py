from bukka.expert_system import solution

decision_tree_solution = solution.Solution(
    name="decision_tree",
    explanation="A decision tree classifier that splits the data based on feature values to make predictions.",
    function_kwargs={
        "criterion": "gini",
        "max_depth": 5,
    },
    function_import="from sklearn.tree import DecisionTreeClassifier",
    function_name="DecisionTreeClassifier",
)

random_forest_solution = solution.Solution(
    name="random_forest",
    explanation="An ensemble of decision trees that improves prediction accuracy and controls overfitting.",
    function_kwargs={
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": 10,
    },
    function_import="from sklearn.ensemble import RandomForestClassifier",
    function_name="RandomForestClassifier",
)

logistic_regression_solution = solution.Solution(
    name="logistic_regression",
    explanation="A logistic regression model for binary or multiclass classification tasks.",
    function_kwargs={
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 100,
    },
    function_import="from sklearn.linear_model import LogisticRegression",
    function_name="LogisticRegression",
)

# Aliases for problem_identifier compatibility
binary_classification = logistic_regression_solution
multi_class_classification = random_forest_solution