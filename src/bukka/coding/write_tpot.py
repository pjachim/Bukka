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
    'binary_classification': 'TPOTClassifier',
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

    def _resolve_tpot_model(self) -> str:
        """Resolve a TPOT class name for the requested problem type."""
        if self.model_type not in TPOT_MAPPING:
            supported_types = ', '.join(sorted(TPOT_MAPPING))
            raise ValueError(
                "TPOT pipeline generation requires a supervised problem type. "
                f"Supported problem types: {supported_types}. "
                f"Got '{self.model_type}'."
            )

        return TPOT_MAPPING[self.model_type]

    def write_tpot_pipeline(self) -> None:
        """Writes the TPOT model class to the specified output path."""
        content = PIPELINE_TEMPLATE.format(tpot_model=self._resolve_tpot_model())
    
        with open(self.file_manager.tpot_pipe_path, 'w') as f:
            f.write(content)