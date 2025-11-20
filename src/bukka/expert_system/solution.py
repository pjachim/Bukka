# Class that stores information about a scikit-learn transformer, this template will be used to generate code for the pipeline.
class Solution:
    """
    A class to represent a scikit-learn transformer or function, used to generate code for a pipeline.

    Parameters
    ----------
    name : str
        The name of the solution.
    function_import : str
        The import statement for the function or transformer.
    function_name : str
        The name of the function or transformer.
    function_kwargs : dict
        A dictionary of keyword arguments to be passed to the function or transformer.
    explanation : str, optional
        A description or explanation of the solution (default is an empty string).

    Attributes
    ----------
    name : str
        The name of the solution.
    function_import : str
        The import statement for the function or transformer.
    function_name : str
        The name of the function or transformer.
    function_kwargs : dict
        A dictionary of keyword arguments to be passed to the function or transformer.
    explanation : str
        A description or explanation of the solution.
    """

    def __init__(self, name: str, function_import: str, function_name: str, function_kwargs: dict, explanation: str = ""):
        self.name = name
        self.function_import = function_import
        self.function_name = function_name
        self.function_kwargs = function_kwargs
        self.explanation = explanation

    def fetch_pipeline_step(self) -> str:
        """
        Generate the code for a pipeline step using the function or transformer.

        Returns
        -------
        str
            A string representing the pipeline step.
        """
        kwargs_str = ", ".join(f"{key}={repr(value)}" for key, value in self.function_kwargs.items())
        return f"{self.name} = {self.function_name}({kwargs_str})"

    def fetch_instantiation(self) -> str:
        """
        Generate the code for instantiating the function or transformer.

        Returns
        -------
        str
            A string representing the instantiation of the function or transformer.
        """
        kwargs_str = ", ".join(f"{key}={repr(value)}" for key, value in self.function_kwargs.items())
        return f"{self.function_name}({kwargs_str})"

    def fetch_import(self) -> str:
        """
        Fetch the import statement for the function or transformer.

        Returns
        -------
        str
            The import statement as a string.
        """
        return self.function_import

    def __repr__(self):
        """
        Return a string representation of the Solution object.

        Returns
        -------
        str
            A string representation of the Solution object.
        """
        return (f"Solution(name={self.name}, function_import={self.function_import}, "
                f"function_name={self.function_name}, function_kwargs={self.function_kwargs}, "
                f"explanation={self.explanation})")