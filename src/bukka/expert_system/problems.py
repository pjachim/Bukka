class Problem:
    def __init__(self, problem_name, description, features: list[str], solutions=[], problem_type: str | None = None):
        self.problem_name = problem_name
        self.description = description
        self.features = features
        self.solutions = solutions
        self.problem_type = problem_type

    def add_solution(self, solution):
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