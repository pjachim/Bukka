import argparse
from bukka.logistics.project import Project
from bukka.utils import bukka_logger

logger = bukka_logger.BukkaLogger(__name__)

def main(name, dataset):
    logger.info('Creating Bukka project!', format_level='h1')
    
    proj = Project(
        name,
        dataset_path=dataset
    )

    proj.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', '-n', type=str, required=True)
    parser.add_argument('--dataset', '-d', type=str, required=False, default=None)

    args = parser.parse_args()

    main(
        name=args.name,
        dataset=args.dataset
    )