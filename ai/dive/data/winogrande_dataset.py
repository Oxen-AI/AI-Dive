from ai.dive.data.dataset import Dataset
import pandas as pd

class WinograndeDataset(Dataset):
    #TODO: add system message for winogrande
    def __init__(self, file, system_msg=""):
        super().__init__()

        self.file = file
        if system_msg == "":
            self.system_msg = "You are an AI assistant. You will be given a sentence with one blank word in it, and two word \
                options to input into that blank. Please output the option that correctly fits based off the context of the sentence."
        else:
            self.system_msg = system_msg

    # For iterating over the dataset
    def __len__(self):
        return len(self.sentence)

    # For iterating over the dataset
    def __getitem__(self, idx):
        #TODO: customize this for winogrande
        qid = self.qid[idx]
        prompt = self.sentence[idx]
        option1 = self.option1[idx]
        option2 = self.option2[idx]


        #diver first gets what is returned here, then does model.predict and appends that, so here, we need to create prompt for model.
        #prompt for model
        modelprompt = f"{self.system_msg}\n\n{prompt}\n\n Options: {option1} {option2}\n\nAnswer:"

        return {
            "qID": qid,
            "system_msg": self.system_msg,
            "prompt": prompt,
            "modelprompt": modelprompt
        }

    # Override this function to load the dataset into memory for fast access
    def _build(self):
#TODO: customize this to fit the winograndetest jsonl
        df = pd.read_json(self.file, lines = True)
        #cols: qID, sentence, option1, option2
        self.qid = df['qID'].tolist()
        self.sentence = df['sentence'].tolist()
        self.option1 = df['option1'].tolist()
        self.option2 = df['option2'].tolist()
        #create new response column, should be answer to question
        #get last integer in qID, if it is 1, answer is option1, if it is 2, answer is option2
        #get last character in qID
        df['response'] = df['qID'].str[-1]
        df['response'] = df['response'].astype(int) #convert to int
        df['response'] = df.apply(lambda row: row['option1'] if row['response'] == 1 else row['option2'], axis=1)
        self.answer = df['response'].tolist()

        #rename sentence column to prompt
        df.rename(columns = {'sentence': 'prompt'}, inplace = True)
        # save df as jsonl titled winogrande_test.jsonl
        df.to_json('winogrande_test.jsonl', orient='records', lines=True)

        # df['response'] = df['response'].str.split('#### ').str[1].str.strip()
        # self.answer = df['response'].tolist()
        # # save df as jsonl titled gsm8k_test.jsonl
        # df.to_json('gsm8k_test.jsonl', orient='records', lines=True)

