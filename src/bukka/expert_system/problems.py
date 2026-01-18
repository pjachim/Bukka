class Problem:
    def __init__(self, problem_name, description, features: list[str], solutions: list | None = None, problem_type: str | None = None):
        self.problem_name = problem_name
        self.description = description
        self.features = features
        self.solutions = solutions if solutions is not None else []
        self.problem_type = problem_type

    def add_solution(self, solution, column_type: str | None = None):
        """Add a solution to the problem if it's compatible with the column type.
        
        Parameters
        ----------
        solution : Solution
            The solution to add to this problem.
        column_type : str | None, optional
            The data type of the column(s) this problem applies to (e.g., 'int', 'float', 'string').
            If None, the solution is added without type checking (default is None).
            
        Examples
        --------
        >>> from bukka.expert_system.problems import Problem
        >>> from bukka.expert_system.solution import Solution
        >>> problem = Problem("Test", "A test problem", ["col1"])
        >>> solution = Solution("imputer", "from sklearn.impute import SimpleImputer",
        ...                     "SimpleImputer", {"strategy": "mean"}, compatible_types=["int", "float"])
        >>> problem.add_solution(solution, column_type="int")
        >>> len(problem.solutions)
        1
        >>> problem.add_solution(solution, column_type="string")  # Won't add, incompatible
        >>> len(problem.solutions)
        1
        """
        if column_type is None or not hasattr(solution, 'compatible_types'):
            # If no column_type specified or solution doesn't have compatible_types, add without checking
            self.solutions += [solution]
        elif column_type in solution.compatible_types:
            # Only add if the solution is compatible with the column type
            self.solutions += [solution]

    def __repr__(self):
        return f"Problem(name={self.problem_name}, description={self.description}, solutions={self.solutions})"
    
    def __getitem__(self, i):
        return self.solutions[i]

class ProblemsToSolve:
    def __init__(self):
        self.problems = []

    def add_problem(self, problem:Problem):
        self.problems += [problem]

    def __getitem__(self, i):
        return self.problems[i]
    
    def __bool__(self):
        return len(self.problems) > 0