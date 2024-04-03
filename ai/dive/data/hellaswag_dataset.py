from ai.dive.data.dataset import Dataset
import pandas as pd

class HellaswagDataset(Dataset):
    #TODO: add system message for Hellaswag
    def __init__(self, file, system_msg=""):
        super().__init__()

        self.file = file
        if system_msg == "":
            self.system_msg = "You are an AI assistant. You will be given a set of sentence(s) that is unfinished, and 4 options to choose \
            from to finish the sentences. The 4 options will be listed as numbers from 0 - 3. Please output the number corresponding to the option \
            that finishes the sentences correctly based on the context."
        else:
            self.system_msg = system_msg

    # For iterating over the dataset
    def __len__(self):
        return len(self.ctx)

    # For iterating over the dataset
    def __getitem__(self, idx):
        #TODO: customize this for hellaswag
        ind = self.ind[idx]
        activity_label = self.activity_label[idx]
        prompt = self.ctx[idx]
        endings = self.endings[idx]


        #diver first gets what is returned here, then does model.predict and appends that, so here, we need to create prompt for model.
        #prompt for model
        modelprompt = f"{self.system_msg}\n\n{prompt}\n\n \
        Options: 0: {endings[0]}\n  1: {endings[1]}\n  2: {endings[2]}\n 3: {endings[3]}\n\nAnswer:"

        return {
            "ind": ind,
            'activity_label': activity_label,
            "system_msg": self.system_msg,
            "prompt": prompt,
            "modelprompt": modelprompt
        }

    # Override this function to load the dataset into memory for fast access
    def _build(self):
#TODO: customize this to fit the hellaswagval jsonl
        df = pd.read_json(self.file, lines = True)
        #cols: ind, activity_label, ctx, endings, label <- answer
        self.ind = df['ind'].tolist()
        self.activity_label = df['activity_label'].tolist()
        self.ctx = df['ctx'].tolist()
        self.endings = df['endings'].tolist()
        self.response = df['label'].tolist()

        # rename label to response
        df.rename(columns = {'label': 'response', "ctx": "prompt"}, inplace = True)
        # save df as jsonl titled hellaswag_validation.jsonl
        df.to_json('hellaswag_validation.jsonl', orient='records', lines=True)


