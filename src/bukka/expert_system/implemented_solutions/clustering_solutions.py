from bukka.expert_system import solution

# Clustering models
kmeans_solution = solution.Solution(
    name="kmeans_clustering",
    explanation="K-Means clustering algorithm to partition data into K distinct clusters based on feature similarity.",
    function_kwargs={
        "n_clusters": 8,
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 300,
    },
    function_import="from sklearn.cluster import KMeans",
    function_name="KMeans",
)

dbscan_solution = solution.Solution(
    name="dbscan_clustering",
    explanation="DBSCAN clustering algorithm that groups together points that are closely packed together, marking points in low-density regions as outliers.",
    function_kwargs={
        "eps": 0.5,
        "min_samples": 5,
    },
    function_import="from sklearn.cluster import DBSCAN",
    function_name="DBSCAN",
)

agglo_clustering_solution = solution.Solution(
    name="agglomerative_clustering",
    explanation="Agglomerative hierarchical clustering that builds nested clusters by merging or splitting them successively.",
    function_kwargs={
        "n_clusters": 2,
        "affinity": "euclidean",
        "linkage": "ward",
    },
    function_import="from sklearn.cluster import AgglomerativeClustering",
    function_name="AgglomerativeClustering",
)

# Alias for problem_identifier compatibility
clustering_analysis = kmeans_solution