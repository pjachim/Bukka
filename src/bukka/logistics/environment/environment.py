import venv
import subprocess
from bukka.logistics.files.file_manager import FileManager
from bukka.utils.reference import requirements

class EnvironmentBuilder:
    """
    Builds and configures a Python virtual environment for a Bukka project.

    This class handles the creation of a virtual environment, installation of
    required packages from a requirements file, and optional editable installation
    of the project itself.

    Parameters
    ----------
    file_manager : FileManager
        Manager for project file paths and directory structure.

    Examples
    --------
    >>> from bukka.logistics.files.file_manager import FileManager
    >>> file_manager = FileManager(project_name="my_project")
    >>> env_builder = EnvironmentBuilder(file_manager)
    >>> env_builder.build_environment()
    """
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager


    def build_environment(self):
        """
        Build the complete Python environment for the project.

        Creates a virtual environment and installs all required packages
        specified in the requirements file.

        Examples
        --------
        >>> env_builder = EnvironmentBuilder(file_manager)
        >>> env_builder.build_environment()
        """
        self._build_venv()
        self._install_packages()

    def _build_venv(self):
        """
        Create a virtual environment with pip included.

        The virtual environment is created at the path specified by
        the file manager's virtual_env attribute.
        """
        venv_client = venv.EnvBuilder(
            with_pip=True
        )

        venv_client.create(
            env_dir=self.file_manager.virtual_env
        )

    def _install_packages(self):
        """
        Write requirements file and install all required packages.

        Creates a requirements.txt file with the standard Bukka dependencies
        and installs them using pip in the virtual environment.
        """
        with open(self.file_manager.requirements_path, 'w') as f:
            f.write(requirements.strip())

        cmd_list = [
            str(self.file_manager.python_path),
            '-m',
            'pip',
            'install',
            '-r',
            str(self.file_manager.requirements_path)
        ]

        subprocess.run(cmd_list)

    def _install_package_editable(self):
        """
        Install the project package in editable mode.

        Performs an editable installation (pip install -e) of the project,
        allowing changes to the source code to be immediately reflected
        without reinstallation.

        Examples
        --------
        >>> env_builder._install_package_editable()
        """
        cmd_list = [
            str(self.file_manager.python_path),
            '-m',
            'pip',
            'install',
            '-e',
            str(self.file_manager.project_path)
        ]

        subprocess.run(cmd_list)