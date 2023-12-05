from ai.dive.data.dataset import Dataset
import os

class DirectoryClassification(Dataset):
    def __init__(self, data_dir):
        super().__init__()

        self.data_dir = data_dir

    # For iterating over the dataset
    def __len__(self):
        return len(self.filepaths)

    # For iterating over the dataset
    def __getitem__(self, idx):
        return {
            "filepath": self.filepaths[idx],
            "class_name": self.labels[idx],
            "filename": self.filepaths[idx].replace(self.data_dir, ""),
        }

    # Override this function to load the dataset into memory for fast access
    def _build(self):
        # iterate over files in directory, taking the directory name as the label
        labels = []
        filepaths = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    labels.append(os.path.basename(root))
                    filepaths.append(os.path.join(root, file))
        self.labels = labels
        self.filepaths = filepaths
