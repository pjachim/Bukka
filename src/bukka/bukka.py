import argparse
from bukka.logistics.project import Project

def main(name, dataset):
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