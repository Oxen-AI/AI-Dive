from ai.dive.data.dataset import Dataset
import json

class LatexDataset(Dataset):
    def __init__(self, file, system_msg=""):
        super().__init__()

        self.file = file
        self.examples = []

    # For iterating over the dataset
    def __len__(self):
        return len(self.examples)

    # For iterating over the dataset
    def __getitem__(self, idx):
        example = self.examples[idx]
        content = example['content']
        
        prompt = f"""The following is the abstract of a research paper titled "{example['title']}" in a stripped LaTeX format:
"{example['abstract']}"

The following is a chunk of the research paper:
---
{content}
---

The above text is an excerpt from a research paper titled "{example['title']}".

Instructions: Write 5 question-anwer pairs about The questions should be an effective test of whether a reader understood the paper, and should be answerable from the content of the paper. Make sure to focus on the parts included in the provided excerpt from the paper. Avoid asking highly generic questions if possible.

Question 1."""
        example['prompt'] = prompt
        return example

    # Override this function to load the dataset into memory for fast access
    def _build(self):
        with open(self.file) as f:
            for line in f:
                self.examples.append(json.loads(line))