import argparse
from bukka.logistics.files.file_manager import FileManager

def main(name, dataset):
    file_manager = FileManager(
        name,
        orig_dataset=dataset
    )

    file_manager.build_skeleton()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', '-n', type=str, required=True)
    parser.add_argument('--dataset', '-d', type=str, required=False, default=None)

    args = parser.parse_args()

    main(
        name=args.name,
        dataset=args.dataset
    )