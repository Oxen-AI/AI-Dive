

from ai.dive.data.dataset import Dataset

import pandas as pd
import os

class FileClassification(Dataset):
    def __init__(
        self,
        data_dir: str,
        file: str,
        path_key: str = 'file',
        label_key: str = 'label'
    ):
        super().__init__()

        self.data_dir = data_dir
        self.path_key = path_key
        self.label_key = label_key
        self.file = os.path.join(data_dir, file)

    # For iterating over the dataset
    def __len__(self):
        return len(self.labels)

    # For iterating over the dataset
    def __getitem__(self, idx):
        return {
            f"filepath": self.full_paths[idx],
            f"{self.path_key}": self.relative_paths[idx],
            f"{self.label_key}": self.labels[idx]
        }

    # Override this function to load the dataset into memory for fast access
    def _build(self):
        df = pd.read_csv(self.file)

        relative_paths = df[self.path_key].tolist()
        full_paths = df[self.path_key].map(lambda x: os.path.join(self.data_dir, x)).tolist()
        labels = df[self.label_key].tolist()

        self.relative_paths = relative_paths
        self.full_paths = full_paths
        self.labels = labels
