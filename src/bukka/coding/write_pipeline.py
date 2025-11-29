from typing import Any
from pathlib import Path
from bukka.coding.utils.template_handler import TemplateBaseClass

# Template for the complete pipeline file
FULL_TEMPLATE = '''
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
{{imports}}

{{instantiations}}

{preprocessor}

pipeline = Pipeline([
    {{pipeline}}
])
'''

PREPROCESSOR_TEMPLATE = '''
preprocessor = ColumnTransformer([
    {transformer_steps}
])
'''

SEPARATOR = ",\n\t"


class PipelineWriter(TemplateBaseClass):
    """Build a pipeline plan from identified problems and chosen solutions.

    The writer takes pipeline steps and generates a complete sklearn Pipeline
    with appropriate imports, instantiations, ColumnTransformer (if needed),
    and final Pipeline definition.

    Parameters
    ----------
    pipeline_steps : list[tuple[Any, Any]]
        List of tuples where each tuple contains (solution_object, problem_object).
        Solution objects must have `fetch_import()`, `fetch_instantiation()`, and
        `name` attributes. Problem objects must have `problem_type` and `features`
        attributes.
    output_path : str | Path
        The file path where the generated pipeline code will be written.

    Attributes
    ----------
    pipeline_steps : list[tuple[Any, Any]]
        The pipeline steps to be processed.
    imports : set[str]
        Set of import statements extracted from solutions.
    instantiations : dict[str, str]
        Dictionary mapping variable names to instantiation code.
    pipeline_definition : str
        The complete generated pipeline code.

    Examples
    --------
    >>> from bukka.expert_system.solution import Solution
    >>> from bukka.expert_system.problems import Problem
    >>> from pathlib import Path
    >>> 
    >>> # Create a transformer solution
    >>> scaler = Solution(
    ...     name="scaler",
    ...     function_import="from sklearn.preprocessing import StandardScaler",
    ...     function_name="StandardScaler",
    ...     function_kwargs={}
    ... )
    >>> 
    >>> # Create a problem for the transformer
    >>> scaling_problem = Problem(
    ...     problem_name="Scaling",
    ...     description="Features need scaling",
    ...     features=["age", "income"],
    ...     solutions=[scaler],
    ...     problem_type="transformer"
    ... )
    >>> 
    >>> # Create pipeline steps
    >>> pipeline_steps = [(scaler, scaling_problem)]
    >>> 
    >>> # Write the pipeline
    >>> writer = PipelineWriter(
    ...     pipeline_steps=pipeline_steps,
    ...     output_path=Path("my_pipeline.py")
    ... )
    >>> writer.write_code()  # Generates and writes the pipeline code

    Notes
    -----
    This class avoids strong typing on solution objects because the
    project's `Solution` wrappers vary; it performs several runtime
    checks to extract import strings, instantiation text and step names.
    """

    def __init__(self, pipeline_steps: list[tuple[Any, Any]], output_path: str | Path) -> None:
        """Create a PipelineWriter.

        Parameters
        ----------
        pipeline_steps : list[tuple[Any, Any]]
            List of (solution_object, problem_object) tuples to include in the pipeline.
        output_path : str | Path
            The file path where the generated pipeline code will be written.

        Examples
        --------
        >>> from pathlib import Path
        >>> writer = PipelineWriter(
        ...     pipeline_steps=[],
        ...     output_path=Path("empty_pipeline.py")
        ... )
        """
        self.pipeline_steps = pipeline_steps if pipeline_steps is not None else []
        
        # Extract all components before initializing parent
        self.imports: set[str] = set()
        self.instantiations: dict[str, str] = {}
        self.pipeline_definition: str = ""
        
        # Build all components
        self._fetch_step_definitions()
        self._fetch_imports()
        
        # Prepare template kwargs
        kwargs = self._build_template_kwargs()
        expected_args = ["imports", "instantiations", "preprocessor", "pipeline"]
        
        # Initialize parent class with template
        super().__init__(
            template=FULL_TEMPLATE,
            output_path=output_path,
            kwargs=kwargs,
            expected_args=expected_args
        )
        
        # Store the pipeline definition for backward compatibility
        self.pipeline_definition = self._fill_template()

    def _fetch_imports(self) -> None:
        """Populate self.imports from the selected pipeline steps.
        
        Examples
        --------
        >>> # Assuming writer has pipeline_steps with solutions
        >>> writer._fetch_imports()
        >>> "from sklearn.pipeline import Pipeline" in writer.imports
        True
        """
        self.imports: set[str] = set()

        for sol_obj, _ in self.pipeline_steps:
            imp = sol_obj.fetch_import()
            if imp:
                self.imports.add(imp)

    def _fetch_step_definitions(self) -> None:
        """Create instantiation lines for each pipeline step.

        The produced strings are simple assignment expressions such as
        ``step_name = SomeTransformer(arg=val)``. The method is defensive
        and supports both small wrapper objects and tuples/lists. Ensures
        unique variable names by appending counters when duplicates are found.
        
        Examples
        --------
        >>> # Assuming writer has pipeline_steps
        >>> writer._fetch_step_definitions()
        >>> "scaler" in writer.instantiations
        True
        >>> "StandardScaler()" in writer.instantiations["scaler"]
        True
        """
        instantiations: dict[str, str] = {}
        used_names: set[str] = set()

        for i, (sol_obj, problem) in enumerate(self.pipeline_steps, start=1):
            # Decide a reasonable variable/name for the step
            if hasattr(sol_obj, "name"):
                name = sol_obj.name
            else:
                name = f"step_{i}"

            # Make a safe python identifier for the variable
            var_name = self.make_python_string_variable_safe(name, lowercase=True)

            # Ensure uniqueness by appending a counter if needed
            counter = 1
            while var_name in used_names:
                var_name = self.make_python_string_variable_safe(f"{var_name}_{counter}", lowercase=True)
                counter += 1
            
            used_names.add(var_name)

            # Instantiate using helper method if available
            inst = sol_obj.fetch_instantiation()

            instantiations[var_name] = inst

        self.instantiations = instantiations

    def _parse_pipeline_steps(self) -> None:
        self.transformers: list[tuple[str, str, list[str]]] = []  # (name, var_name, columns)
        self.manipulators: list[str] = []  # var_names for multi-column steps
        self.model_step: str | None = None

        for (sol_obj, problem), var_name in zip(self.pipeline_steps, self.instantiations.keys()):
            problem_type = getattr(problem, "problem_type", None)
            features = getattr(problem, "features", [])
            
            if problem_type == "transformer":
                # Single-column transformer
                self.transformers.append((var_name, var_name, features))
            elif problem_type == "manipulator":
                # Multi-column manipulator
                self.manipulators.append(var_name)
            elif problem_type == "model":
                # Final model step
                self.model_step = var_name
            else:
                # Default: treat as manipulator if no type specified
                self.manipulators.append(var_name)

    def _build_preprocessor(self, transformers, pipeline_steps) -> tuple[str, list[str]]:
        """Build the ColumnTransformer preprocessor code if needed.

        This method constructs the ColumnTransformer definition based on
        the pipeline steps that are transformers. It groups single-column
        transformers into the ColumnTransformer and leaves multi-column
        manipulators as separate pipeline steps.

        Returns
        -------
        str
            The ColumnTransformer definition code, or an empty string if
            no transformers are present.

        Examples
        --------
        >>> # Assuming writer has pipeline_steps with transformers
        >>> preprocessor_code = writer._build_preprocessor()
        >>> "ColumnTransformer" in preprocessor_code
        True
        """
        if transformers:
            # Create ColumnTransformer for single-column transformers
            ct_items = []
            for name, var_name, columns in transformers:
                cols_repr = repr(columns) if len(columns) > 1 else repr(columns[0]) if columns else "[]"
                ct_items.append(f"('{name}', {var_name}, {cols_repr})")
            
            ct_body = SEPARATOR.join(ct_items)
            PREPROCESSOR_TEMPLATE.strip().format(

            )
            preprocessor_str = f"preprocessor = ColumnTransformer([\n\t\t{ct_body}\n\t], remainder='passthrough')"
            pipeline_steps.append("('preprocessor', preprocessor)")

            transformer_steps

        else:
            preprocessor_str = ""

    def _build_template_kwargs(self) -> dict[str, Any]:
        """Build the kwargs dictionary for template substitution.
        
        Constructs all sections of the pipeline file: imports, instantiations,
        preprocessor (ColumnTransformer), and final pipeline definition.
        
        Returns
        -------
        dict[str, Any]
            Dictionary with keys 'imports', 'instantiations', 'preprocessor',
            and 'pipeline' containing the formatted code sections.
        
        Examples
        --------
        >>> # Assuming writer is properly initialized
        >>> kwargs = writer._build_template_kwargs()
        >>> "imports" in kwargs
        True
        >>> "pipeline" in kwargs
        True
        """
        # Format imports
        imports_str = "\n".join(sorted(self.imports)) if self.imports else ""
        
        # Format instantiations
        instantiation_lines = [
            f"{var_name} = {inst}" 
            for var_name, inst in self.instantiations.items()
        ]
        instantiations_str = "\n".join(instantiation_lines) if instantiation_lines else ""
        
        # Categorize steps by type

        
        # Build preprocessor (ColumnTransformer)
        preprocessor_str = ""
        pipeline_steps: list[str] = []

        self._build_preprocessor(transformers, pipeline_steps)


        # Add manipulators to pipeline
        for var_name in manipulators:
            pipeline_steps.append(f"('{var_name}', {var_name})")

        # Add model as final step
        if model_step:
            pipeline_steps.append(f"('{model_step}', {model_step})")

        # Create final pipeline
        if pipeline_steps:
            pipeline_body = ",\n\t".join(pipeline_steps)
            pipeline_str = f"pipeline = Pipeline([\n\t{pipeline_body}\n])"
        else:
            pipeline_str = "pipeline = Pipeline([])"
        
        return {
            "imports": imports_str,
            "instantiations": instantiations_str,
            "preprocessor": preprocessor_str,
            "pipeline": pipeline_str
        }
