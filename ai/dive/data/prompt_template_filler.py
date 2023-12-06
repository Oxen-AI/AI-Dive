from ai.dive.data.dataset import Dataset
import pandas as pd
from langchain.prompts import PromptTemplate

class PromptTemplateFiller(Dataset):
    def __init__(self, file, template, num_repeats):
        super().__init__()

        self.file = file
        self.template = template
        self.num_repeats = num_repeats
        self.prompt_key = 'response'

    # For iterating over the dataset
    def __len__(self):
        return self.num_repeats

    # For iterating over the dataset
    def __getitem__(self, idx):
        prompt_template = PromptTemplate.from_template(self.template)
        prompt = self.prompts[idx]
        completed_template = prompt_template.format(prompt=prompt)
        print(prompt)
        
        return {
            "idx": idx,
            "input": prompt,
            "prompt": completed_template
        }

    # Override this function to load the dataset into memory for fast access
    def _build(self):
        df = pd.read_csv(self.file)
        self.prompts = df[self.prompt_key].tolist()