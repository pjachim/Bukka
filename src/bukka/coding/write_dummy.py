from bukka.utils.files.file_manager import FileManager

PIPELINE_TEMPLATE = '''
from sklearn.dummy import {dummy_model} as dummy_model
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('model', dummy_model())
])'''

DUMMY_MAPPING = {
    'binary_classification': 'DummyClassifier',
    'multiclass_classification': 'DummyClassifier',
    'regression': 'DummyRegressor',
}

class DummyWriter:
    """
    Generates and writes a dummy model class for testing purposes.

    Attributes:
        model_type (str): The type of dummy model to generate (e.g., 'DummyClassifier', 'DummyRegressor').
        output_path (str): The file path where the generated class will be written.

    Methods:
        write_dummy_class(): Writes the dummy model class to the specified output path.
    """
    def __init__(self, file_manager: FileManager, model_type: str | None = None) -> None:
        self.file_manager = file_manager
        self.model_type = model_type

    def write_dummy_class(self) -> None:
        """Writes the dummy model class to the specified output path."""
        content = PIPELINE_TEMPLATE.format(dummy_model=DUMMY_MAPPING[self.model_type])
        
        with open(self.file_manager.dummy_pipe_path, 'w') as f:
            f.write(content)