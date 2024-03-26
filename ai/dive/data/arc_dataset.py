from ai.dive.data.dataset import Dataset
import pandas as pd

class ARCDataset(Dataset):
    def __init__(self, file, system_msg=""):
        super().__init__()

        self.file = file
        if system_msg == "":
            self.system_msg = "You are an AI assistant, you will be given a multiple choice question and 4 answer choices A) B) C) D). Output the correct answer choice only specifying the letter corresponding to the answer A,B,C,D."
        else:
            self.system_msg = system_msg

    # For iterating over the dataset
    def __len__(self):
        return len(self.prompts)

    # For iterating over the dataset
    def __getitem__(self, idx):
        id = self.id[idx]
        prompt = self.prompts[idx]
        choices = self.choices[idx]
        answer_idx = self.answer_idx[idx]

        prompt = f"{self.system_msg}\n\n{prompt}\n\nChoices:\n\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n\nAnswer:"

        # convert 1,2,3,4 to A,B,C,D
        answer = chr(answer_idx + 65)

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
        self.choices = df['choices'].tolist()
        self.answer_idx = df['answer_idx'].tolist()