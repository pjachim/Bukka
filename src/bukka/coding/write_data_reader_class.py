from bukka.utils.files.file_manager import FileManager

CLASS_TEMPLATE = '''
import narwhals as nw
from config import DATAFRAME_BACKEND, TRAIN_DATASET_PATH, TEST_DATASET_PATH
from narwhals.typing import FrameT

class DataReader:
    """
    A class to read training and testing data from Parquet files using Narwhals.

    Attributes:
        train_filepath (str): The file path to the training data Parquet file.
        test_filepath (str): The file path to the testing data Parquet file.

    Methods:
        read_train_data(): Reads and returns the training data as a Narwhals DataFrame.
        read_test_data(): Reads and returns the testing data as a Narwhals DataFrame.
    """
    def __init__(self, train_filepath: str = TRAIN_DATASET_PATH, test_filepath: str = TEST_DATASET_PATH):
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath
        self.dataframe_backend = DATAFRAME_BACKEND

    def read_train_data(self) -> FrameT:
        """Reads the training data from the training Parquet file."""
        return self._read_file(self.train_filepath)

    def read_test_data(self) -> FrameT:
        """Reads the testing data from the testing Parquet file."""
        return self._read_file(self.test_filepath)

    def _read_file(self, filepath: str) -> FrameT:
        """Reads a Parquet file and returns a Narwhals DataFrame."""
        return nw.read_parquet(filepath, backend=self.dataframe_backend)

'''

# These methods are added only if a target column is specified (no need for X/y split for unsupervised tasks)
ADDITIONAL_SUPERVISED_METHODS = '''
    def readXy_train(self, target_column: str | None = {target_column}) -> tuple[FrameT, FrameT]:
        """Reads the training data and splits it into features and target."""
        return self._readXy(self.train_filepath, is_train=True, target_column=target_column)

    def readXy_test(self, target_column: str | None = {target_column}) -> tuple[FrameT, FrameT]:
        """Reads the testing data and splits it into features and target."""
        return self._readXy(self.test_filepath, is_train=False, target_column=target_column)

    def _readXy(self, filepath: str, is_train: bool, target_column: str | None = {target_column}) -> tuple[FrameT, FrameT]:
        """Reads a Parquet file and splits it into features and target."""
        df = self._read_file(filepath)
        X = df.drop([target_column])
        y = df.select(nw.col(target_column))
        return X, y
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
    def __init__(self, file_manager: FileManager, target_column: str | None = None) -> None:
        self.file_manager = file_manager
        self.target_column = target_column

    def write_code(self) -> None:
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
        """
        if self.target_column is not None:
            class_template = CLASS_TEMPLATE + ADDITIONAL_SUPERVISED_METHODS
        else:
            class_template = CLASS_TEMPLATE

        filled_template = class_template.strip()
        # Use relative paths from project root
        train_rel = self.file_manager.train_data_file.relative_to(self.file_manager.project_path)
        test_rel = self.file_manager.test_data_file.relative_to(self.file_manager.project_path)
        filled_template = filled_template.format(
            target_column=repr(self.target_column)
        )
        return filled_template