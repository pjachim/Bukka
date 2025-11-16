import venv
import os
import subprocess
from bukka.logistics.files.file_manager import FileManager
from bukka.utils.reference import requirements

class EnvironmentBuilder:
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager


    def build_environment(self):
        self._build_venv()
        self._install_packages()

    def _build_venv(self):
        venv_client = venv.EnvBuilder(
            with_pip=True
        )

        venv_client.create(
            env_dir=self.file_manager.virtual_env
        )

    def _install_packages(self):
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
