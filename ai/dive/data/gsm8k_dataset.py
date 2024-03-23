from ai.dive.data.dataset import Dataset
import pandas as pd

class GSM8KDataset(Dataset):
    def __init__(self, file, system_msg=""):
        super().__init__()

        self.file = file
        if system_msg == "":
            self.system_msg = " You are an AI assistant, you will be given a grade school math problem, find the answer to the problem, then output 'Answer: ', then the answer as a single number with no dollar signs or decimals in it."
        else:
            self.system_msg = system_msg

    # For iterating over the dataset
    def __len__(self):
        return len(self.prompts)

    # For iterating over the dataset
    def __getitem__(self, idx):
        id = self.id[idx]
        prompt = self.prompts[idx]


        #diver first gets what is returned here, then does model.predict and appends that, so here, we need to create prompt for model.
        #prompt for model
        modelprompt = f"{self.system_msg}\n\n{prompt}\n\nAnswer:"

        return {
            "id": id,
            "system_msg": self.system_msg,
            "prompt": prompt,
            "modelprompt": modelprompt
        }

    # Override this function to load the dataset into memory for fast access
    def _build(self):

        df = pd.read_json(self.file, lines = True)
        self.id = df['id'].tolist()
        self.prompts = df['prompt'].tolist()
        #create new answer column, which is the string after #### 
        df['response'] = df['response'].str.split('#### ').str[1].str.strip()
        self.answer = df['response'].tolist()
        # save df as jsonl titled gsm8k_test.jsonl
        df.to_json('gsm8k_test.jsonl', orient='records', lines=True)

