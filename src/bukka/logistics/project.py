from bukka.logistics.files.file_manager import FileManager
from bukka.logistics.environment.environment import EnvironmentBuilder

class Project:
    def __init__(self, name, dataset_path):
        self.name = name
        self.dataset_path = dataset_path

    def run(self):
        self._build_skeleton()
        self._setup_environment()

    def _build_skeleton(self):
        self.file_manager = FileManager(
            project_path=self.name,
            orig_dataset=self.dataset_path
        )

        self.file_manager.build_skeleton()

    def _setup_environment(self):
        self.environ_manager = EnvironmentBuilder(
            file_manager=self.file_manager
        )

        self.environ_manager.build_environment()