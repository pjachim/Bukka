import pytest

from bukka.expert_system.solution import Solution
from bukka.expert_system.problems import Problem, ProblemsToSolve


class TestSolution:
    def test_fetch_methods_and_repr(self):
        sol = Solution(
            name="my_step",
            function_import="from pkg import Fn",
            function_name="Fn",
            function_kwargs={"a": 1, "b": "x"},
            explanation="explain",
        )

        # Import string is returned as-is
        assert sol.fetch_import() == "from pkg import Fn"

        # Instantiation and pipeline step formatting
        inst = sol.fetch_instantiation()
        assert "Fn(" in inst and "a=1" in inst and "b='x'" in inst

        step = sol.fetch_pipeline_step()
        assert step.startswith("my_step = Fn(") or step.startswith("my_step = Fn")

        # repr contains the class name and provided name
        r = repr(sol)
        assert "Solution(" in r and "my_step" in r


class TestProblemsContainer:
    def test_problem_and_problems_to_solve(self):
        p = Problem(problem_name="Nulls", description="has nulls", features=["col1"], solutions=[], problem_type="transformer")
        # add a solution placeholder
        p.add_solution("sol_func")
        assert p[0] == "sol_func"

        # repr contains the problem name
        assert "Nulls" in repr(p)

        container = ProblemsToSolve()
        assert not bool(container)
        container.add_problem(p)
        assert bool(container)
        assert container[0].problem_name == "Nulls"
