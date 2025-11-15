from pathlib import Path
import shutil

class FileManager:
    def __init__(
            self,
            project_path: str,
            orig_dataset: str
        ) -> None:

        self.project_path = Path(project_path)
        self.orig_dataset = Path(orig_dataset)
        self._build_paths()
    
    def _build_paths(self) -> None:
        self.data_path = self.project_path / 'data'
        self.train_data = self.data_path / 'train'
        self.test_data = self.data_path / 'test'

        self.pipes = self.project_path / 'pipelines'
        self.generated_pipes = self.pipes / 'generated'
        self.baseline_pipes = self.pipes / 'baseline'
        self.candidate_pipes = self.pipes / 'candidate'

        self.virtual_env = self.project_path / '.venv'
        self.scripts = self.project_path / 'scripts'

        self.dataset_path = self.data_path / self.orig_dataset.name

        # Files (not created with skeleton)
        self.starter_notebook_path = self.project_path / 'starter.ipynb'
        self.requirements_path = self.project_path / 'requirements.txt'
        self.readme_path = self.project_path / 'README.md'
        self.gitignore_path = self.project_path / '.gitignore'
        
        
    def _make_path(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def build_skeleton(self) -> None:
        self._make_path(self.data_path)
        self._make_path(self.train_data)
        self._make_path(self.test_data)

        self._make_path(self.pipes)
        (self.pipes / '__init__.py').touch()
        self._make_path(self.generated_pipes)
        (self.generated_pipes / '__init__.py').touch()
        self._make_path(self.baseline_pipes)
        (self.baseline_pipes / '__init__.py').touch()
        self._make_path(self.candidate_pipes)
        (self.candidate_pipes / '__init__.py').touch()

        self._make_path(self.virtual_env)
        self._make_path(self.scripts)
        (self.scripts / '__init__.py').touch()

        shutil.copy2(self.orig_dataset, self.dataset_path)

