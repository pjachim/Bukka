from bukka.logistics.files.file_manager import FileManager

class_template = '''
import polars as pl

class DataReader:
    """
    A class to read training and testing data from Parquet files using Polars.

    Attributes:
        train_filepath (str): The file path to the training data Parquet file.
        test_filepath (str): The file path to the testing data Parquet file.

    Methods:
        read_train_data(): Reads and returns the training data as a Polars DataFrame.
        read_test_data(): Reads and returns the testing data as a Polars DataFrame.
    """
    def __init__(self, train_filepath: str = {train_filepath}, test_filepath: str = {test_filepath}):
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath

    def read_train_data(self):
        """Reads the training data from the training Parquet file."""
        return self._read_file(self.train_filepath)

    def read_test_data(self):
        """Reads the testing data from the testing Parquet file."""
        return self._read_file(self.test_filepath)

    def _read_file(self, filepath: str):
        """Reads a Parquet file and returns a Polars DataFrame."""
        return pl.read_parquet(filepath)
'''


class DataReaderWriter:
    """
    Generates and writes a DataReader class for loading train/test parquet files.

    This class constructs a Python source file containing a `DataReader` class
    with pre-configured file paths for training and testing datasets.

    Parameters
    ----------
    file_handler : FileHandler
        Handler providing paths to data files and the target output location.

    Examples
    --------
    >>> from bukka.logistics.files.file_manager import FileHandler
    >>> file_handler = FileHandler(project_name="my_project")
    >>> writer = DataReaderWriter(file_handler)
    >>> writer.write_class()  # Writes DataReader class to file
    """
    def __init__(self, file_manager: FileManager) -> None:
        self.file_manager = file_manager

    def write_class(self) -> None:
        """
        Write the DataReader class to the configured output path.

        Generates Python source code from the template and writes it to
        the file specified by `file_manager.data_reader_path`.

        Examples
        --------
        >>> writer = DataReaderWriter(file_manager)
        >>> writer.write_class()
        """
        class_code = self._fill_template()
        with open(self.file_manager.data_reader_path, 'w') as file:
            file.write(class_code)

    def _fill_template(self) -> str:
        """
        Fill the DataReader template with train and test file paths.

        Returns
        -------
        str
            Python source code for the DataReader class with paths substituted.

        Examples
        --------
        >>> writer = DataReaderWriter(file_manager)
        >>> code = writer._fill_template()
        >>> assert "train_filepath" in code
        """
        filled_template = class_template.strip()
        # Use relative paths from project root
        train_rel = self.file_manager.train_data_file.relative_to(self.file_manager.project_path)
        test_rel = self.file_manager.test_data_file.relative_to(self.file_manager.project_path)
        filled_template = filled_template.format(
            train_filepath=repr(str(train_rel).replace('\\', '/')),
            test_filepath=repr(str(test_rel).replace('\\', '/'))
        )
        return filled_template