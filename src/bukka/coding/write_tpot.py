from bukka.utils.files.file_manager import FileManager

PIPELINE_TEMPLATE = '''
import tpot
from sklearn.pipeline import Pipeline

automl_model = tpot.{tpot_model}(
    preprocessing=True
)

# Putting this in a pipeline for faster itteration.
pipeline = Pipeline([
    ('model', automl_model)
])'''

TPOT_MAPPING = {
    'classification': 'TPOTClassifier',
    'multiclass_classification': 'TPOTClassifier',
    'regression': 'TPOTRegressor',
}

class TPOTWriter:
    """
    Generates and writes a TPOT model class for prototyping purposes.

    Attributes:
        model_type (str): The type of TPOT model to generate (e.g., 'TPOTClassifier', 'TPOTRegressor').
        output_path (str): The file path where the generated class will be written.

    Methods:
        write_tpot_class(): Writes the TPOT model class to the specified output path.
    """
    def __init__(self, file_manager: FileManager, model_type: str | None = None) -> None:
        self.file_manager = file_manager
        self.model_type = model_type

    def write_tpot_pipeline(self) -> None:
        """Writes the TPOT model class to the specified output path."""
        content = PIPELINE_TEMPLATE.format(tpot_model=TPOT_MAPPING[self.model_type])
        write_path = self.file_manager.generated_pipes / 'tpot_pipe.py'