from ai.dive.data.dataset import Dataset
import pandas as pd

class GSM8KDataset(Dataset):
    def __init__(self, file, system_msg=""):
        super().__init__()

        self.file = file
        if system_msg == "":
            self.system_msg = "You are an AI assistant, you will be given a grade school math problem, output with a single number that is the correct answer to the problem."
        else:
            self.system_msg = system_msg

    # For iterating over the dataset
    def __len__(self):
        return len(self.prompts)

    # For iterating over the dataset
    def __getitem__(self, idx):
        id = self.id[idx]
        prompt = self.prompts[idx]
        answer = self.answer[idx]

        prompt = f"{self.system_msg}\n\n{prompt}\n\nAnswer:"

        return {
            "id": id,
            "prompt": prompt,
            "answer": answer
        }

    # Override this function to load the dataset into memory for fast access
    def _build(self):
        df = pd.read_json(self.file, lines = True)
        self.id = df['id'].tolist()
        self.prompts = df['prompt'].tolist()
        #create new answer column, which is the string after ####
        df['answer'] = df['response'].str.split('#### ').str[1].str.strip()
        self.answer = df['answer'].tolist()
