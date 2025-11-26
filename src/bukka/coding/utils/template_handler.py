from pathlib import Path
from typing import Any

class TemplateBaseClass:
    """
    Base class for handling template-based code generation.
    
    This class provides functionality to fill string templates with provided arguments
    and write the resulting code to a file. It uses Python's str.format() method to
    substitute placeholders in the template with actual values.
    
    Attributes
    ----------
    template : str
        The template string containing placeholders for formatting.
    output_path : str | Path
        The file path where the generated code will be written.
    kwargs : dict[str, Any]
        Dictionary of keyword arguments to fill the template placeholders.
    expected_args : list[str]
        List of expected argument names for the template.
    
    Examples
    --------
    >>> from pathlib import Path
    >>> template = '''
    ... class {class_name}:
    ...     def __init__(self):
    ...         self.value = {default_value}
    ... '''
    >>> kwargs = {'class_name': 'MyClass', 'default_value': 42}
    >>> handler = TemplateBaseClass(
    ...     template=template,
    ...     output_path='output.py',
    ...     kwargs=kwargs,
    ...     expected_args=['class_name', 'default_value']
    ... )
    >>> handler.write_class()  # Writes the filled template to 'output.py'
    
    >>> # Using Path objects
    >>> output_path = Path('generated') / 'my_class.py'
    >>> handler = TemplateBaseClass(
    ...     template=template,
    ...     output_path=output_path,
    ...     kwargs={'class_name': 'AnotherClass', 'default_value': 100}
    ... )
    >>> filled = handler._fill_template()
    >>> print(filled)
    class AnotherClass:
        def __init__(self):
            self.value = 100
    """
    def __init__(
            self,
            template: str,
            output_path: str | Path,
            kwargs: dict[str, Any],
            expected_args: list[str] | None = None
        ) -> None:
        """
        Initialize the template handler.
        
        Parameters
        ----------
        template : str
            The template string containing placeholders in the format {placeholder_name}.
        output_path : str | Path
            The file path where the generated code will be written.
        kwargs : dict[str, Any]
            Dictionary mapping placeholder names to their replacement values.
        expected_args : list[str] | None, optional
            List of expected argument names for validation purposes. If None, an empty
            list is used. Default is None.
        
        Examples
        --------
        >>> template = "def {func_name}(): return {value}"
        >>> handler = TemplateBaseClass(
        ...     template=template,
        ...     output_path='func.py',
        ...     kwargs={'func_name': 'get_answer', 'value': 42}
        ... )
        """
        self.template = template
        self.output_path = output_path
        self.kwargs = kwargs

        if expected_args is None:
            self.expected_args = []
        else:
            self.expected_args = list(expected_args)

    def write_class(self) -> None:
        """
        Fill the template and write the resulting code to the output file.
        
        This method fills the template with the provided kwargs and writes the
        resulting code to the file specified in output_path. The file is created
        if it doesn't exist, or overwritten if it does.
        
        Examples
        --------
        >>> template = "x = {value}"
        >>> handler = TemplateBaseClass(
        ...     template=template,
        ...     output_path='config.py',
        ...     kwargs={'value': 123}
        ... )
        >>> handler.write_class()
        >>> # File 'config.py' now contains: "x = 123"
        """
        class_code = self._fill_template()

        with open(self.output_path, 'w') as file:
            file.write(class_code)

    def _fill_template(self) -> str:
        """
        Fill the template with provided kwargs and return the formatted string.
        
        This private method strips leading/trailing whitespace from the template,
        then uses Python's str.format() method to substitute all placeholders with
        their corresponding values from kwargs.
        
        Returns
        -------
        str
            The template string with all placeholders filled with corresponding values.
        
        Examples
        --------
        >>> template = "Hello {name}, you are {age} years old."
        >>> handler = TemplateBaseClass(
        ...     template=template,
        ...     output_path='greeting.txt',
        ...     kwargs={'name': 'Alice', 'age': 30}
        ... )
        >>> filled = handler._fill_template()
        >>> print(filled)
        Hello Alice, you are 30 years old.
        """
        filled_template = self.template.strip()
        filled_template = filled_template.format(**self.kwargs)

        return filled_template