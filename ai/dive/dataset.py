# Base class for all datasets

class Dataset:
    def __init__(self):
        pass

    def len(self):
        raise NotImplementedError("Dataset must implement len()")
    
    def item_at(self, idx):
        raise NotImplementedError("Dataset must implement item_at()")

    def load(self):
        raise NotImplementedError("Dataset must implement load()")
    
