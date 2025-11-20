from bukka.expert_system import solution

onehotencoder_solution = solution.Solution(
    name="one_hot_encoder",
    explanation="Applies One-Hot Encoding to convert categorical variables into a format that can be provided to ML algorithms.",
    function_kwargs={
        "sparse": False,
        "handle_unknown": "ignore"
    },
    function_import="from sklearn.preprocessing import OneHotEncoder",
    function_name="OneHotEncoder",
)