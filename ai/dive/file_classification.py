

from ai.dive.dataset import Dataset

import pandas as pd
import os

class FileClassification(Dataset):
    def __init__(
        self,
        data_dir: str,
        file: str,
        path_key: str = 'file',
        label_key: str = 'label',
    ):
        super().__init__()

        self.data_dir = data_dir
        self.path_key = path_key
        self.label_key = label_key
        self.file = os.path.join(data_dir, file)

    # For iterating over the dataset
    def len(self):
        return len(self.labels)
    
    # For iterating over the dataset
    def item_at(self, idx):
        return {
            "filename": self.paths[idx], 
            "class_name": self.labels[idx]
        }

    # Override this function to load the dataset
    def load(self):
        df = pd.read_csv(self.file)
        image_paths, labels = self.load_data_from_df(df)
        self.labels = labels
        self.paths = image_paths

    def load_data_from_df(self, df: pd.DataFrame):
        # Map the paths to the full path
        image_paths = df[self.path_key].map(lambda x: os.path.join(self.data_dir, x)).tolist()
        labels = df[self.label_key].tolist()
        return image_paths, labels