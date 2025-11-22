import argparse
from bukka.logistics.project import Project
from bukka.utils import bukka_logger

logger = bukka_logger.BukkaLogger(__name__)


def main(name: str, dataset: str | None, target: str | None) -> None:
    """Create a Bukka project, set it up, and generate a candidate pipeline.

    Args:
        name: Project name / path to create.
        dataset: Path to the original dataset file (optional).
        target: Name of the target column (optional; pass None for clustering).
    """
    logger.info("Creating Bukka project!", format_level="h1")

    proj = Project(
        name,
        dataset_path=dataset,
        target_column=target
    )

    proj.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', '-n', type=str, required=True)
    parser.add_argument('--dataset', '-d', type=str, required=False, default=None)
    parser.add_argument('--target', '-t', type=str, required=False, default=None)

    args = parser.parse_args()

    main(
        name=args.name,
        dataset=args.dataset,
        target=args.target
    )
