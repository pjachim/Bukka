from typing import Any, List, Set, Tuple, Dict
from bukka.expert_system.problem_identifier import ProblemIdentifier
import random
from bukka.expert_system.solution import Solution


class PipelineWriter:
    """Build a pipeline plan from identified problems and chosen solutions.

    The writer selects one solution for each identified problem and one
    final ML solution (e.g. a model or clustering step). It then prepares
    import statements, instantiation code for each step and a minimal
    sklearn `Pipeline` definition string.

    Notes
    -----
    This class avoids strong typing on solution objects because the
    project's `Solution` wrappers vary; it performs several runtime
    checks to extract import strings, instantiation text and step names.
    """

    def __init__(self, problem_identifier: ProblemIdentifier) -> None:
        """Create a `PipelineWriter`.

        Args:
            problem_identifier: An instance of `ProblemIdentifier` that
                exposes `problems_to_solve` and `ml_problem` with
                candidate `solutions`.
        """
        self.problem_identifier = problem_identifier
        # Prefer random ordering for the ml_problem solutions so pop() is non-deterministic
        try:
            random.shuffle(self.problem_identifier.ml_problem.solutions)
        except Exception:
            # If ml_problem.solutions is not shufflable, ignore
            pass

        # Public results filled by `write()`
        self.pipeline_steps: List[Any] = []
        self.imports: Set[str] = set()
        self.instantiations: Dict[str, str] = {}
        self.pipeline_definition: str = ""

    def write(self) -> Tuple[List[Any], Set[str]]:
        """Assemble the pipeline plan and return steps and imports.

        Returns
        -------
        tuple[list, set]
            A tuple of (`pipeline_steps`, `imports`). `pipeline_steps` is
            a list of selected solution objects (shape depends on project
            conventions). `imports` is a set of import statement strings.
        """
        self._arrange_pipeline()
        self._fetch_step_definitions()
        self._fetch_imports()
        self._define_sklearn_pipeline()
        return self.pipeline_steps, self.imports

    def _arrange_pipeline(self) -> None:
        """Select processors for each identified problem and the ML step.

        The selected items are stored in `self.pipeline_steps` in order.
        """
        self.pipeline_steps: list[Solution] = []
        # Select one processor per problem (if any)
        self.pipeline_steps += self._processor_selection()

        # Append the ML solution (if available). Use pop() to follow the
        # original intention of selecting one solution from the shuffled list.
        try:
            ml_solutions = getattr(self.problem_identifier.ml_problem, "solutions", [])
            if ml_solutions:
                self.pipeline_steps.append(ml_solutions.pop())
        except Exception:
            pass

    def _processor_selection(self) -> list[Solution]:
        """Choose one solution per problem.

        Returns a list of chosen solution objects (exact shape depends on
        how problems/solutions are represented in the expert system).
        """
        chosen: list[Solution] = []
        problems = getattr(self.problem_identifier.problems_to_solve, "problems", [])
        for p in problems:
            try:
                sol_list = getattr(p, "solutions", [])
                if sol_list:
                    chosen.append(random.choice(sol_list))
            except Exception:
                # Skip problematic entries rather than crash
                continue
        return chosen

    def _fetch_imports(self) -> None:
        """Populate `self.imports` from the selected pipeline steps."""
        imports: Set[str] = set()
        for step in self.pipeline_steps:
            imp = step.fetch_import()
            if imp:
                imports.add(imp)
        self.imports = imports

    def _fetch_step_definitions(self) -> None:
        """Create instantiation lines for each pipeline step.

        The produced strings are simple assignment expressions such as
        ``step_name = SomeTransformer(arg=val)``. The method is defensive
        and supports both small wrapper objects and tuples/lists.
        """
        instantiations: dict[str, str] = {}
        used_names: set[str] = set()

        for idx, sol_obj in enumerate(self.pipeline_steps, start=1):
            # Decide a reasonable variable/name for the step
            name = None
            if hasattr(sol_obj, "name"):
                name = sol_obj.name
            else:
                name = f"step_{idx}"

            # Make a safe python identifier for the variable
            var_name = str(name).lower().replace(" ", "_")

            # Ensure uniqueness by appending a counter if needed
            original_var_name = var_name
            counter = 1
            while var_name in used_names:
                var_name = f"{original_var_name}_{counter}"
                counter += 1
            
            used_names.add(var_name)

            # Instantiate using helper method if available
            inst = sol_obj.fetch_instantiation()

            instantiations[var_name] = inst

        self.instantiations = instantiations

    def _define_sklearn_pipeline(self) -> None:
        """Generate a small string representing a sklearn Pipeline creation.

        The string includes the instantiation lines followed by a
        `Pipeline([...])` assignment using the variable names from
        `self.instantiations`.
        """
        # Ensure step definitions are available
        if not self.instantiations:
            self._fetch_step_definitions()

        # Build pipeline tuple list from dict keys
        pipeline_items = []
        for var_name in self.instantiations.keys():
            pipeline_items.append(f"('{var_name}', {var_name})")

        pipeline_body = ",\n\t".join(pipeline_items)

        lines: List[str] = []
        if self.imports:
            # Add imports as provided (these are strings, likely full import lines)
            for imp in sorted(self.imports):
                lines.append(imp)

        # Add instantiation lines
        for var_name, inst in self.instantiations.items():
            lines.append(f"{var_name} = {inst}")

        # Add the pipeline construction. We don't import Pipeline here to avoid
        # coupling; the import string should be present in `self.imports` if needed.
        lines.append(f"\npipeline = Pipeline([\n\t{pipeline_body}\n])")

        self.pipeline_definition = "\n".join(lines)
