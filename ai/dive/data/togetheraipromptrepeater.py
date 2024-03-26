from ai.dive.data.dataset import Dataset
import pandas as pd

class togetheraipromptrepeater(Dataset):
    def __init__(self, file, template):
        super().__init__()

        self.file = file
        self.template = template #this will be a dictionary

    # For iterating over the dataset
    def __len__(self):
        return len(self.prompts)

    # For iterating over the dataset
    def __getitem__(self, idx):
        id = self.id[idx]
        prompt = self.prompts[idx]
        choices = self.choices[idx]
        answeridx = self.answeridx[idx]
        
        return {
            "id": id,
            "prompt": prompt,
            'choices': choices,
            "answeridx": answeridx
        }

    # Override this function to load the dataset into memory for fast access
    def _build(self):
        df = pd.read_json(self.file, lines = True)
        self.id = df['id'].tolist()
        self.prompts = df['prompt'].tolist()
        self.choices = df['choices'].tolist()
        self.answeridx = df['answer_idx'].tolist()