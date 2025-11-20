import argparse
from bukka.logistics.project import Project
from bukka.data_management.dataset import Dataset
from bukka.expert_system.problem_identifier import ProblemIdentifier
from bukka.utils import bukka_logger

logger = bukka_logger.BukkaLogger(__name__)

def main(name, dataset, target):
    logger.info('Creating Bukka project!', format_level='h1')
    
    proj = Project(
        name,
        dataset_path=dataset
    )

    proj.run()

    # identify problems in the dataset
    dataset = Dataset(
        target,
        proj.file_manager.dataset_path
    )

    problem_identifier = ProblemIdentifier(dataset)
    problem_identifier.multivariate_problems()
    problem_identifier.univariate_problems()

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
