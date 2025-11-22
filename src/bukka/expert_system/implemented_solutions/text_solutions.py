# Solutions for working with text data in the expert system.
from bukka.expert_system import solution

tfidf_solution = solution.Solution(
    name="tfidf_vectorizer",
    explanation="Applies TF-IDF vectorization to convert text data into numerical format suitable for machine learning models.",
    function_kwargs={
        "max_features": 1000,
        "ngram_range": (1, 2),
        "stop_words": "english"
    },
    function_import="from sklearn.feature_extraction.text import TfidfVectorizer",
    function_name="TfidfVectorizer",
)

countvectorizer_solution = solution.Solution(
    name="count_vectorizer",
    explanation="Applies Count Vectorization to convert text data into numerical format by counting word occurrences.",
    function_kwargs={
        "max_features": 1000,
        "ngram_range": (1, 2),
        "stop_words": "english"
    },
    function_import="from sklearn.feature_extraction.text import CountVectorizer",
    function_name="CountVectorizer",
)

hashingvectorizer_solution = solution.Solution(
    name="hashing_vectorizer",
    explanation="Applies Hashing Vectorization to convert text data into numerical format using a hashing trick.",
    function_kwargs={
        "n_features": 1000,
        "ngram_range": (1, 2),
        "alternate_sign": False
    },
    function_import="from sklearn.feature_extraction.text import HashingVectorizer",
    function_name="HashingVectorizer",
)