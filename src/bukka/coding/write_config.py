CONFIG_TEMPLATE = """
DATAFRAME_BACKEND = '{backend_name}'"""

class ConfigWriter:
    """
    Generates and writes a configuration file for a Bukka project.

    This class constructs a Python source file containing configuration settings
    for the Bukka project, such as the DataFrame backend to use.

    Parameters
    ----------
    output_path : str
        The file path where the configuration file will be written.
    backend_name : str
        The name of the DataFrame backend to use (e.g., 'pandas', 'polars').

    Examples
    --------
    >>> writer = ConfigWriter(output_path="config.py", backend_name="pandas")
    >>> writer.write_config()  # Writes the config file with pandas backend
    """
    def __init__(self, output_path: str, backend_name: str):
        self.output_path = output_path
        self.backend_name = backend_name

    def write_config(self) -> None:
        """
        Write the configuration file to the configured output path.

        Generates Python source code from the template and writes it to
        the specified file.

        Examples
        --------
        >>> writer = ConfigWriter(output_path="config.py", backend_name="polars")
        >>> writer.write_config()  # Writes the config file with polars backend
        """
        config_code = CONFIG_TEMPLATE.format(backend_name=self.backend_name)
        with open(self.output_path, 'w') as file:
            file.write(config_code)