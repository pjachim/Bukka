from typing import Any
from bukka.expert_system.problem_identifier import ProblemIdentifier
from bukka.expert_system.solution import Solution
from bukka.expert_system.problems import Problem

import random

class PipelineBuilder:
    """Build a pipeline plan from identified problems and chosen solutions.

    The builder selects one solution for each identified problem and one
    final ML solution (e.g. a model or clustering step). It then prepares
    import statements, instantiation code for each step and a minimal
    sklearn `Pipeline` definition string.

    Notes
    -----
    This class avoids strong typing on solution objects because the
    project's `Solution` wrappers vary; it performs several runtime
    checks to extract import strings, instantiation text and step names.
    """

    def __init__(self, dataset, target_column) -> None:
        """Create a `PipelineBuilder`.

        Args:
            problem_identifier: An instance of `ProblemIdentifier` that
                exposes `problems_to_solve` and `ml_problem` with
                candidate `solutions`.
        """
        self.problem_identifier = ProblemIdentifier(
            dataset, target_column
        )
        self.problem_identifier.identify_problems()
        
        # Prefer random ordering for the ml_problem solutions so pop() is non-deterministic
        if hasattr(self.problem_identifier, 'ml_problem') and self.problem_identifier.ml_problem is not None:
            try:
                solutions = getattr(self.problem_identifier.ml_problem, 'solutions', [])
                if solutions:
                    random.shuffle(solutions)
            except Exception:
                # If ml_problem.solutions is not shufflable, ignore
                pass
    
    def build_pipeline(self) -> None:
        """Select processors for each identified problem and the ML step.

        The selected items are stored in `self.pipeline_steps` as tuples
        of (solution, problem) to preserve problem metadata.
        """
        self.pipeline_steps: list[tuple[Solution, Any]] = []
        # Select one processor per problem (if any)
        self.pipeline_steps += self._processor_selection()

        # Append the ML solution (if available). Use pop() to follow the
        # original intention of selecting one solution from the shuffled list.
        if hasattr(self.problem_identifier, 'ml_problem') and self.problem_identifier.ml_problem is not None:
            try:
                ml_solutions = getattr(self.problem_identifier.ml_problem, "solutions", [])
                if ml_solutions:
                    self.pipeline_steps.append((ml_solutions.pop(), self.problem_identifier.ml_problem))
            except Exception as e:
                # Log but don't crash if ml_problem is malformed
                pass

        return self.pipeline_steps

    def _processor_selection(self) -> list[tuple[list[Solution], Any]]:
        """Choose one solution per problem.

        Returns a list of (solution, problem) tuples to preserve problem
        metadata like features and problem_type.
        """
        chosen: list[tuple[Solution, Problem]] = []
        problems = getattr(self.problem_identifier.problems_to_solve, "problems", [])
        for p in problems:
            try:
                sol_list = getattr(p, "solutions", [])
                if sol_list:
                    chosen.append((random.choice(sol_list), p))
            except Exception:
                # Skip problematic entries rather than crash
                continue
        return chosen