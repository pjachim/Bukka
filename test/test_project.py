"""Unit tests for the Project class."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from bukka.project import Project


class TestProjectInitialization:
    """Test suite for Project class initialization."""

    def test_project_initialization_with_minimal_args(self):
        """Test Project initialization with only required arguments.
        
        Examples
        --------
        >>> proj = Project(name="test_proj")
        >>> assert proj.name == "test_proj"
        >>> assert proj.dataset_path is None
        """
        proj = Project(name="test_project")
        
        assert proj.name == "test_project"
        assert proj.dataset_path is None
        assert proj.target_column is None
        assert proj.file_manager is None
        assert proj.environ_manager is None
        assert proj.skip_venv is False
        assert proj.backend == "polars"
        assert proj.problem_type == "auto"
        assert proj.train_size == 0.8
        assert proj.stratify is True
        assert proj.strata is None

    def test_project_initialization_with_dataset(self):
        """Test Project initialization with dataset path.
        
        Examples
        --------
        >>> proj = Project(
        ...     name="my_proj",
        ...     dataset_path="/path/to/data.csv",
        ...     target_column="target"
        ... )
        >>> assert proj.dataset_path == "/path/to/data.csv"
        >>> assert proj.target_column == "target"
        """
        proj = Project(
            name="my_project",
            dataset_path="/path/to/data.csv",
            target_column="label"
        )
        
        assert proj.name == "my_project"
        assert proj.dataset_path == "/path/to/data.csv"
        assert proj.target_column == "label"

    def test_project_initialization_with_custom_backend(self):
        """Test Project initialization with custom backend.
        
        Examples
        --------
        >>> proj = Project(name="proj", backend="pyarrow")
        >>> assert proj.backend == "pyarrow"
        """
        proj = Project(
            name="test_project",
            backend="pyarrow"
        )
        
        assert proj.backend == "pyarrow"

    def test_project_initialization_with_skip_venv(self):
        """Test Project initialization with skip_venv enabled.
        
        Examples
        --------
        >>> proj = Project(name="proj", skip_venv=True)
        >>> assert proj.skip_venv is True
        """
        proj = Project(
            name="test_project",
            skip_venv=True
        )
        
        assert proj.skip_venv is True

    def test_project_initialization_with_custom_train_size(self):
        """Test Project initialization with custom train_size.
        
        Examples
        --------
        >>> proj = Project(name="proj", train_size=0.7)
        >>> assert proj.train_size == 0.7
        """
        proj = Project(
            name="test_project",
            train_size=0.7
        )
        
        assert proj.train_size == 0.7

    def test_project_initialization_with_stratification_options(self):
        """Test Project initialization with stratification options.
        
        Examples
        --------
        >>> proj = Project(name="proj", stratify=True, strata=["col1"])
        >>> assert proj.stratify is True
        >>> assert proj.strata == ["col1"]
        """
        proj = Project(
            name="test_project",
            stratify=True,
            strata=["feature1", "feature2"]
        )
        
        assert proj.stratify is True
        assert proj.strata == ["feature1", "feature2"]

    def test_project_initialization_with_problem_type(self):
        """Test Project initialization with specific problem type.
        
        Examples
        --------
        >>> proj = Project(name="proj", problem_type="binary_classification")
        >>> assert proj.problem_type == "binary_classification"
        """
        proj = Project(
            name="test_project",
            problem_type="regression"
        )
        
        assert proj.problem_type == "regression"


class TestProjectBuildSkeleton:
    """Test suite for Project._build_skeleton method."""

    def test_build_skeleton_creates_file_manager(self):
        """Test that _build_skeleton initializes FileManager.
        
        Examples
        --------
        >>> proj = Project(name="test_proj")
        >>> proj._build_skeleton()
        >>> assert proj.file_manager is not None
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_path = Path(tmp_dir) / "test_project"
            proj = Project(name=str(project_path))
            
            assert proj.file_manager is None
            proj._build_skeleton()
            assert proj.file_manager is not None

    @patch('bukka.project.FileManager')
    def test_build_skeleton_calls_file_manager_build(self, mock_file_manager_class):
        """Test that _build_skeleton calls FileManager.build_skeleton.
        
        Examples
        --------
        >>> from unittest.mock import patch
        >>> with patch('bukka.project.FileManager') as mock_fm:
        ...     proj = Project(name="test")
        ...     proj._build_skeleton()
        ...     mock_fm.return_value.build_skeleton.assert_called_once()
        """
        mock_fm_instance = MagicMock()
        mock_file_manager_class.return_value = mock_fm_instance
        
        proj = Project(name="test_project", dataset_path="data.csv")
        proj._build_skeleton()
        
        # Verify FileManager was instantiated with correct arguments
        mock_file_manager_class.assert_called_once_with(
            project_path="test_project",
            orig_dataset="data.csv"
        )
        
        # Verify build_skeleton was called
        mock_fm_instance.build_skeleton.assert_called_once()


class TestProjectSetupEnvironment:
    """Test suite for Project._setup_environment method."""

    @patch('bukka.project.EnvironmentBuilder')
    def test_setup_environment_creates_environ_manager(self, mock_env_builder_class):
        """Test that _setup_environment initializes EnvironmentBuilder.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> with patch('bukka.project.EnvironmentBuilder') as mock_eb:
        ...     proj = Project(name="test")
        ...     proj.file_manager = MagicMock()
        ...     proj._setup_environment()
        ...     assert proj.environ_manager is not None
        """
        mock_env_instance = MagicMock()
        mock_env_builder_class.return_value = mock_env_instance
        
        proj = Project(name="test_project")
        proj.file_manager = MagicMock()
        
        assert proj.environ_manager is None
        proj._setup_environment()
        assert proj.environ_manager is not None

    @patch('bukka.project.EnvironmentBuilder')
    def test_setup_environment_calls_build_environment(self, mock_env_builder_class):
        """Test that _setup_environment calls build_environment.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> with patch('bukka.project.EnvironmentBuilder') as mock_eb:
        ...     proj = Project(name="test")
        ...     proj.file_manager = MagicMock()
        ...     proj._setup_environment()
        ...     mock_eb.return_value.build_environment.assert_called_once()
        """
        mock_env_instance = MagicMock()
        mock_env_builder_class.return_value = mock_env_instance
        
        proj = Project(name="test_project")
        proj.file_manager = MagicMock()
        
        proj._setup_environment()
        mock_env_instance.build_environment.assert_called_once()

    def test_setup_environment_raises_error_if_file_manager_is_none(self):
        """Test that _setup_environment raises RuntimeError if FileManager is None.
        
        Examples
        --------
        >>> proj = Project(name="test")
        >>> try:
        ...     proj._setup_environment()
        ... except RuntimeError as e:
        ...     assert "FileManager must be initialized" in str(e)
        """
        proj = Project(name="test_project")
        
        with pytest.raises(RuntimeError) as exc_info:
            proj._setup_environment()
        
        assert "FileManager must be initialized" in str(exc_info.value)


class TestProjectRun:
    """Test suite for Project.run method integration."""

    @patch('bukka.project.EnvironmentBuilder')
    @patch('bukka.project.FileManager')
    def test_run_without_dataset_skips_pipeline_generation(
        self, mock_file_manager_class, mock_env_builder_class
    ):
        """Test that run() skips pipeline generation when no dataset is provided.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> with patch('bukka.project.FileManager'), patch('bukka.project.EnvironmentBuilder'):
        ...     proj = Project(name="test", dataset_path=None)
        ...     proj.run()
        ...     # Pipeline generation should be skipped
        """
        mock_fm_instance = MagicMock()
        mock_file_manager_class.return_value = mock_fm_instance
        mock_env_instance = MagicMock()
        mock_env_builder_class.return_value = mock_env_instance
        
        proj = Project(name="test_project", dataset_path=None)
        proj.run()
        
        # Verify skeleton was built
        mock_fm_instance.build_skeleton.assert_called_once()
        
        # Verify environment was set up (since skip_venv=False by default)
        mock_env_instance.build_environment.assert_called_once()

    @patch('bukka.project.StarterNotebookWriter')
    @patch('bukka.project.ConfigWriter')
    @patch('bukka.project.DataReaderWriter')
    @patch('bukka.project.PipelineWriter')
    @patch('bukka.project.PipelineBuilder')
    @patch('bukka.project.Dataset')
    @patch('bukka.project.EnvironmentBuilder')
    @patch('bukka.project.FileManager')
    def test_run_with_dataset_generates_pipeline(
        self, mock_file_manager_class, mock_env_builder_class,
        mock_dataset_class, mock_pipeline_builder_class, mock_pipeline_writer_class,
        mock_data_reader_writer_class, mock_config_writer_class, mock_notebook_writer_class
    ):
        """Test that run() generates pipeline when dataset is provided.
        
        Examples
        --------
        >>> from unittest.mock import patch, MagicMock
        >>> # Multiple patches required for complete test
        >>> proj = Project(name="test", dataset_path="data.csv", target_column="target")
        >>> proj.run()
        >>> # Pipeline, config, and notebook should be generated
        """
        # Setup mocks
        mock_fm_instance = MagicMock()
        mock_fm_instance.generated_pipes = Path("/tmp/pipelines")
        mock_fm_instance.config_path = Path("/tmp/config.py")
        mock_file_manager_class.return_value = mock_fm_instance
        
        mock_env_instance = MagicMock()
        mock_env_builder_class.return_value = mock_env_instance
        
        mock_dataset_instance = MagicMock()
        mock_dataset_class.return_value = mock_dataset_instance
        
        mock_builder_instance = MagicMock()
        mock_builder_instance.build_pipeline.return_value = []
        mock_pipeline_builder_class.return_value = mock_builder_instance
        
        mock_writer_instance = MagicMock()
        mock_pipeline_writer_class.return_value = mock_writer_instance
        
        mock_data_reader_instance = MagicMock()
        mock_data_reader_writer_class.return_value = mock_data_reader_instance
        
        mock_config_instance = MagicMock()
        mock_config_writer_class.return_value = mock_config_instance
        
        mock_notebook_instance = MagicMock()
        mock_notebook_writer_class.return_value = mock_notebook_instance
        
        # Create and run project
        proj = Project(
            name="test_project",
            dataset_path="data.csv",
            target_column="target"
        )
        proj.run()
        
        # Verify skeleton was built
        mock_fm_instance.build_skeleton.assert_called_once()
        
        # Verify environment was set up
        mock_env_instance.build_environment.assert_called_once()
        
        # Verify dataset was created
        mock_dataset_class.assert_called_once()
        
        # Verify pipeline was built and written
        mock_builder_instance.build_pipeline.assert_called_once()
        mock_writer_instance.write_code.assert_called_once()
        
        # Verify data reader was written
        mock_data_reader_instance.write_code.assert_called_once()
        
        # Verify config was written
        mock_config_instance.write_config.assert_called_once()
        
        # Verify notebook was written
        mock_notebook_instance.write_notebook.assert_called_once()

    @patch('bukka.project.FileManager')
    def test_run_with_skip_venv_skips_environment_setup(self, mock_file_manager_class):
        """Test that run() skips environment setup when skip_venv=True.
        
        Examples
        --------
        >>> from unittest.mock import patch
        >>> with patch('bukka.project.FileManager'):
        ...     proj = Project(name="test", skip_venv=True)
        ...     proj.run()
        ...     # Environment setup should be skipped
        """
        mock_fm_instance = MagicMock()
        mock_file_manager_class.return_value = mock_fm_instance
        
        proj = Project(name="test_project", skip_venv=True)
        proj.run()
        
        # Verify skeleton was built
        mock_fm_instance.build_skeleton.assert_called_once()
        
        # Verify environ_manager was never created
        assert proj.environ_manager is None

class TestProjectMLflowIntegration:
    """Test suite for Project MLflow integration."""
    
    def test_project_initialization_with_mlflow_enabled(self):
        """Test Project initialization with MLflow enabled."""
        proj = Project(
            name="test_project",
            enable_mlflow=True
        )
        
        assert proj.enable_mlflow is True
        assert proj.mlflow_tracking_uri is None
    
    def test_project_initialization_with_mlflow_and_custom_uri(self):
        """Test Project initialization with MLflow and custom tracking URI."""
        proj = Project(
            name="test_project",
            enable_mlflow=True,
            mlflow_tracking_uri="http://localhost:5000"
        )
        
        assert proj.enable_mlflow is True
        assert proj.mlflow_tracking_uri == "http://localhost:5000"
    
    @patch('bukka.project.EnvironmentBuilder')
    @patch('bukka.project.FileManager')
    def test_environment_builder_receives_mlflow_flag(
        self,
        mock_file_manager_class,
        mock_env_builder_class
    ):
        """Test that EnvironmentBuilder receives enable_mlflow flag."""
        mock_fm = MagicMock()
        mock_file_manager_class.return_value = mock_fm
        
        proj = Project(name="test", enable_mlflow=True)
        proj.run()
        
        # Verify EnvironmentBuilder was called with enable_mlflow=True
        mock_env_builder_class.assert_called_once_with(
            file_manager=mock_fm,
            enable_mlflow=True
        )
    
    @patch('bukka.project.ConfigWriter')
    @patch('bukka.project.StarterNotebookWriter')
    @patch('bukka.project.DataReaderWriter')
    @patch('bukka.project.PipelineWriter')
    @patch('bukka.project.Dataset')
    @patch('bukka.project.EnvironmentBuilder')
    @patch('bukka.project.PyprojectTomlWriter')
    @patch('bukka.project.FileManager')
    def test_write_mlflow_setup_called_when_enabled(
        self,
        mock_fm_class,
        mock_toml_class,
        mock_env_class,
        mock_dataset_class,
        mock_pipeline_class,
        mock_data_reader_class,
        mock_notebook_class,
        mock_config_class
    ):
        """Test that _write_mlflow_setup is called when MLflow is enabled."""
        # Setup mocks
        mock_fm = MagicMock()
        mock_fm.mlruns_path = MagicMock()
        mock_fm.mlruns_path.mkdir = MagicMock()
        mock_fm_class.return_value = mock_fm
        
        # Create a temporary dataset file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            dataset_path = f.name
        
        try:
            proj = Project(
                name="test",
                dataset_path=dataset_path,
                target_column="target",
                enable_mlflow=True
            )
            
            # Patch the _write_mlflow_setup and _write_mlflow_notebook methods to verify they're called
            with patch.object(proj, '_write_mlflow_setup') as mock_mlflow_setup, \
                 patch.object(proj, '_write_mlflow_notebook') as mock_mlflow_notebook:
                proj.run()
                
                # Verify both methods were called
                mock_mlflow_setup.assert_called_once()
                mock_mlflow_notebook.assert_called_once()
        finally:
            # Cleanup
            Path(dataset_path).unlink(missing_ok=True)
    
    @patch('bukka.project.ConfigWriter')
    @patch('bukka.project.StarterNotebookWriter')
    @patch('bukka.project.DataReaderWriter')
    @patch('bukka.project.PipelineWriter')
    @patch('bukka.project.Dataset')
    @patch('bukka.project.EnvironmentBuilder')
    @patch('bukka.project.PyprojectTomlWriter')
    @patch('bukka.project.FileManager')
    def test_write_mlflow_setup_not_called_when_disabled(
        self,
        mock_fm_class,
        mock_toml_class,
        mock_env_class,
        mock_dataset_class,
        mock_pipeline_class,
        mock_data_reader_class,
        mock_notebook_class,
        mock_config_class
    ):
        """Test that _write_mlflow_setup is NOT called when MLflow is disabled."""
        # Setup mocks
        mock_fm = MagicMock()
        mock_fm_class.return_value = mock_fm
        
        # Create a temporary dataset file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            dataset_path = f.name
        
        try:
            proj = Project(
                name="test",
                dataset_path=dataset_path,
                target_column="target",
                enable_mlflow=False
            )
            
            # Patch the _write_mlflow_setup method to verify it's NOT called
            with patch.object(proj, '_write_mlflow_setup') as mock_mlflow_setup:
                proj.run()
                
                # Verify _write_mlflow_setup was NOT called
                mock_mlflow_setup.assert_not_called()
        finally:
            # Cleanup
            Path(dataset_path).unlink(missing_ok=True)
    
    @patch('bukka.coding.write_mlflow_setup.MLflowSetupWriter')
    @patch('bukka.project.FileManager')
    def test_write_mlflow_setup_creates_directory(
        self,
        mock_fm_class,
        mock_mlflow_writer_class
    ):
        """Test that _write_mlflow_setup creates mlruns directory."""
        mock_fm = MagicMock()
        mock_mlruns_path = MagicMock()
        mock_fm.mlruns_path = mock_mlruns_path
        mock_fm_class.return_value = mock_fm
        
        proj = Project(name="test", enable_mlflow=True)
        proj.file_manager = mock_fm
        proj._write_mlflow_setup()
        
        # Verify mlruns directory creation was called
        mock_mlruns_path.mkdir.assert_called_once_with(exist_ok=True)