from bukka.expert_system.data_management.dataset import Dataset
from bukka.expert_system.data_management.problems import Problems

class SchemaConstructor:
    def __init__(self, dataset:Dataset):
        self.dataset = dataset
        self.problems = Problems

    def construct(self):
        for feature in self.dataset.feature_columns:
            feature

    def _identify_type(self):
        ...

    def _identify(self):
        ...