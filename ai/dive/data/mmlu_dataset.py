from ai.dive.data.dataset import Dataset
import pandas as pd

class MMLUDataset(Dataset):
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
        prompt = self.prompts[idx]
        choices = self.choices[idx]
        answer = self.answer[idx]

        prompt = f"{self.system_msg}\n\n{prompt}\n\nChoices:\n\nA) {choices['A']}\nB) {choices['B']}\nC) {choices['D']}\nD) {choices['D']}\n\nAnswer:\n\n"

        return {
            "prompt": prompt,
            "choices": choices,
            "answer": answer
        }

    # Override this function to load the dataset into memory for fast access
    def _build(self):
        df = pd.read_json(self.file, lines = True)
        self.prompts = df['prompt'].tolist()
        self.choices = df['choices'].tolist()
        self.answer = df['answer'].tolist()