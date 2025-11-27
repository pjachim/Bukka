from typing import Any, List, Set, Tuple, Dict
from bukka.expert_system.problem_identifier import ProblemIdentifier
from pathlib import Path

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

    def __init__(self, pipeline_steps: List[Tuple[Any, Any]], output_path: str | Path) -> None:
        """Create a `PipelineWriter`.

        Args:
            problem_identifier: An instance of `ProblemIdentifier` that
                exposes `problems_to_solve` and `ml_problem` with
                candidate `solutions`.
        """
        self.pipeline_steps = pipeline_steps
        self.output_path = Path(output_path)

        # Public results filled by `write()`
        self.imports: Set[str] = set()
        self.instantiations: Dict[str, str] = {}
        self.pipeline_definition: str = ""

    def write(self) -> None:
        """Assemble the pipeline plan and return steps and imports.

        Returns
        -------
        tuple[list, set]
            A tuple of (`pipeline_steps`, `imports`). `pipeline_steps` is
            a list of selected solution objects (shape depends on project
            conventions). `imports` is a set of import statement strings.
        """
        self._fetch_step_definitions()
        self._fetch_imports()
        self._define_sklearn_pipeline()

        with open(self.output_path, "w") as f:
            f.write(self.pipeline_definition)

    def _fetch_imports(self) -> None:
        """Populate `self.imports` from the selected pipeline steps."""
        imports: Set[str] = set()
        for sol_obj, _ in self.pipeline_steps:
            imp = sol_obj.fetch_import()
            if imp:
                imports.add(imp)
        # Add ColumnTransformer and Pipeline imports
        imports.add("from sklearn.compose import ColumnTransformer")
        imports.add("from sklearn.pipeline import Pipeline")
        self.imports = imports

    def _fetch_step_definitions(self) -> None:
        """Create instantiation lines for each pipeline step.

        The produced strings are simple assignment expressions such as
        ``step_name = SomeTransformer(arg=val)``. The method is defensive
        and supports both small wrapper objects and tuples/lists.
        """
        instantiations: dict[str, str] = {}
        used_names: set[str] = set()

        for idx, (sol_obj, problem) in enumerate(self.pipeline_steps, start=1):
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
        """Generate sklearn ColumnTransformer and Pipeline code.

        Separates transformers (single-column operations), manipulators
        (multi-column operations), and models. Creates a ColumnTransformer
        for column-specific steps, then wraps everything in a Pipeline.
        """
        # Ensure step definitions are available
        if not self.instantiations:
            self._fetch_step_definitions()

        # Categorize steps by type
        transformers: List[Tuple[str, str, List[str]]] = []  # (name, var_name, columns)
        manipulators: List[str] = []  # var_names for multi-column steps
        model_step: str | None = None

        for (sol_obj, problem), var_name in zip(self.pipeline_steps, self.instantiations.keys()):
            problem_type = getattr(problem, "problem_type", None)
            features = getattr(problem, "features", [])
            
            if problem_type == "transformer":
                # Single-column transformer
                transformers.append((var_name, var_name, features))
            elif problem_type == "manipulator":
                # Multi-column manipulator
                manipulators.append(var_name)
            elif problem_type == "model":
                # Final model step
                model_step = var_name
            else:
                # Default: treat as manipulator if no type specified
                manipulators.append(var_name)

        lines: List[str] = []
        if self.imports:
            # Add imports
            for imp in sorted(self.imports):
                lines.append(imp)

        lines.append("")  # Blank line after imports

        # Add instantiation lines
        for var_name, inst in self.instantiations.items():
            lines.append(f"{var_name} = {inst}")

        lines.append("")  # Blank line before pipeline construction

        # Build the pipeline
        pipeline_steps: List[str] = []

        if transformers:
            # Create ColumnTransformer for single-column transformers
            ct_items = []
            for name, var_name, columns in transformers:
                cols_repr = repr(columns) if len(columns) > 1 else repr(columns[0]) if columns else "[]"
                ct_items.append(f"('{name}', {var_name}, {cols_repr})")
            
            ct_body = ",\n\t\t".join(ct_items)
            lines.append(f"preprocessor = ColumnTransformer([\n\t\t{ct_body}\n\t], remainder='passthrough')")
            lines.append("")
            pipeline_steps.append("('preprocessor', preprocessor)")

        # Add manipulators to pipeline
        for var_name in manipulators:
            pipeline_steps.append(f"('{var_name}', {var_name})")

        # Add model as final step
        if model_step:
            pipeline_steps.append(f"('{model_step}', {model_step})")

        # Create final pipeline
        if pipeline_steps:
            pipeline_body = ",\n\t".join(pipeline_steps)
            lines.append(f"pipeline = Pipeline([\n\t{pipeline_body}\n])")
        else:
            lines.append("pipeline = Pipeline([])")

        self.pipeline_definition = "\n".join(lines)
