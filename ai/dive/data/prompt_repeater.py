from ai.dive.data.dataset import Dataset

class PromptRepeater(Dataset):
    def __init__(self, prompt, num_repeats):
        super().__init__()

        self.prompt = prompt
        self.num_repeats = num_repeats

    # For iterating over the dataset
    def __len__(self):
        return self.num_repeats

    # For iterating over the dataset
    def __getitem__(self, idx):
        return {
            "idx": idx,
            "prompt": self.prompt
        }

    # Override this function to load the dataset into memory for fast access
    def _build(self):
        pass