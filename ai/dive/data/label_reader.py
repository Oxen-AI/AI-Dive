

class LabelReader:
    def __init__(self, labels_file=None):
        self.labels_file = labels_file
        self.label_indices = {}
        self._build()

    # Load model into memory
    def _build(self):
        print("Loading labels...")
        if self.labels_file is None:
            raise Exception("You must provide a labels file")
        else:
            with open(self.labels_file, 'r') as f:
                self.class_labels = f.read().splitlines()
                for i, label in enumerate(self.class_labels):
                    self.label_indices[label] = i
        print(f"Got {len(self.class_labels)} labels")

    def num_labels(self):
        return len(self.class_labels)
    
    def label_at(self, index):
        return self.class_labels[index]
    
    def index_of(self, label):
        return self.label_indices[label]
    
    def labels(self):
        return self.class_labels