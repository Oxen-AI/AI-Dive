from ai.dive.data.dataset import Dataset
from ai.dive.prompts.assistant_prompt import AssistantPrompt
import json

class PromptResponseDataset(Dataset):
    def __init__(self, file, n_shot_file=None):
        super().__init__()

        self.file = file
        self.examples = []

    # For iterating over the dataset
    def __len__(self):
        return len(self.examples)

    # For iterating over the dataset
    def __getitem__(self, idx):
        example = self.examples[idx]
        question = example['prompt']
        prompt = AssistantPrompt(example, should_add_answer=False).render()
        example['instruction'] = question
        example['prompt'] = prompt
        return example

    # Override this function to load the dataset into memory for fast access
    def _build(self):
        with open(self.file) as f:
            for line in f:
                self.examples.append(json.loads(line))