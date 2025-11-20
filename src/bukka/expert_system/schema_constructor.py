from bukka.data_management.dataset import Dataset
from bukka.expert_system.problems import ProblemsToSolve, Problem

# Class that scans the dataset and identifies problems based on attributes in the dataset. 
class ProblemIdentifier:
    def __init__(self, dataset:Dataset):
        self.dataset = dataset
        self.problems_to_solve = ProblemsToSolve()

    def construct(self):
        for feature in self.dataset.feature_columns:
            feature

    def _identify_type(self):
        ...

    def _identify(self):
        ...