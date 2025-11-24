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

standardize_categories_solution = solution.Solution(
    name="standardize_categories",
    explanation="Standardizes inconsistent categorical values (e.g., fixes case, trims whitespace, corrects typos).",
    function_kwargs={},
    function_import="from bukka.preprocessing.categorical import standardize_categories",
    function_name="standardize_categories",
)

encode_categories_solution = solution.Solution(
    name="encode_categories",
    explanation="Encodes categorical values using ordinal or label encoding.",
    function_kwargs={
        "encoding_type": "ordinal"
    },
    function_import="from bukka.preprocessing.categorical import encode_categories",
    function_name="encode_categories",
)

# For ProblemIdentifier compatibility
standardize_categories = standardize_categories_solution
encode_categories = encode_categories_solution