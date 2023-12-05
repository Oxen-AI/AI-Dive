# Base class for all datasets

class Dataset:
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError("Dataset must implement __len__()")

    def __getitem__(self, idx):
        raise NotImplementedError("Dataset must implement __getitem__()")

    def _build(self):
        raise NotImplementedError("Implementing _build() for dataset may make it faster")

    def build(self):
        try:
            self._build()
        except NotImplementedError:
            print(f"Warning _build() not implemented for dataset.")

