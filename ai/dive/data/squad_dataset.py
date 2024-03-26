from ai.dive.data.dataset import Dataset
import pandas as pd

class SQuADDataset(Dataset):
    def __init__(self, file, system_msg=""):
        super().__init__()

        self.file = file
        if system_msg == "":
            self.system_msg = "Read the context and question and extract the answer."
        else:
            self.system_msg = system_msg

    # For iterating over the dataset
    def __len__(self):
        return len(self.prompts)

    # For iterating over the dataset
    def __getitem__(self, idx):
        id = self.id[idx]
        question = self.prompts[idx]
        answers = self.answers[idx]
        context = self.context[idx]

        prompt = f"{self.system_msg}\n\nContext:\n\n{context}\n\nQuestion:\n\n{question}\n\nAnswer:\n\n"

        return {
            "id": id,
            "prompt": prompt,
            "question": question,
            "context": context,
            "answers": answers,
        }

    # Override this function to load the dataset into memory for fast access
    def _build(self):
        df = pd.read_json(self.file, lines = True)
        self.id = df['id'].tolist()
        self.prompts = df['prompt'].tolist()
        self.context = df['context'].tolist()
        self.answers = df['answers'].tolist()