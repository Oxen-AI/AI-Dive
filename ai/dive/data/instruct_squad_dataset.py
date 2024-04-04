from ai.dive.data.dataset import Dataset
from ai.dive.prompts.qa_prompt import QAPrompt
from ai.dive.prompts.assistant_prompt import AssistantPrompt
import json

class InstructSquadDataset(Dataset):
    def __init__(self, file, n_shot_file=None):
        super().__init__()

        self.file = file
        self.n_shot_file = n_shot_file
        self.examples = []

    # For iterating over the dataset
    def __len__(self):
        return len(self.examples)

    # For iterating over the dataset
    def __getitem__(self, idx):
        example = self.examples[idx]
        question = example['prompt']
        prompt = QAPrompt(example, n_shot_examples=self.n_shot_examples, should_add_answer=False, completion_name="Bessie:").render()
        prompt = AssistantPrompt({'prompt': prompt}, should_add_answer=False).render().strip()
        example['question'] = question
        example['prompt'] = prompt
        return example

    # Override this function to load the dataset into memory for fast access
    def _build(self):
        with open(self.file) as f:
            for line in f:
                self.examples.append(json.loads(line))

        self.n_shot_examples = []
        if self.n_shot_file is not None:
            with open(self.n_shot_file) as f:
                for line in f:
                    self.n_shot_examples.append(json.loads(line))