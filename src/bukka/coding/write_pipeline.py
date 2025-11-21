from bukka.expert_system.problem_identifier import ProblemIdentifier
import random

class PipelineWriter:
    def __init__(self, problem_identifier: ProblemIdentifier) -> None:
        self.problem_identifier = problem_identifier
        random.shuffle(self.problem_identifier.ml_problem.solutions)

    def write(self) -> str:
        self._arrange_pipeline()
        self._fetch_imports()
        return self.pipeline_steps, self.imports

    def _arrange_pipeline(self):
        self.pipeline_steps = []
        self.pipeline_steps += self._processor_selection()
        self.pipeline_steps += [self.problem_identifier.ml_problem.solutions.pop()]

    def _processor_selection(self):
        return [random.choice(p.solutions) for p in self.problem_identifier.problems_to_solve.problems]

    def _fetch_imports(self):
        self.imports = set([p.solution[0].import_statement for p in self.pipeline_steps])

    def _fetch_step_definitions(self):
        self.instantiations = []

        for step in self.pipeline_steps:
            solution = step.solution[0]

            args = []
            for arg, value in solution.function_kwargs.items():
                if isinstance(value, str):
                    args.append(f"{arg}='{value}'")
                else:
                    args.append(f"{arg}={value}")

        self.imports = set([p.solution[0].import_statement for p in self.pipeline_steps])

    def _define_sklearn_pipeline(self):
        ...