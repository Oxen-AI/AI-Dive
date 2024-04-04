
from ai.dive.models.model import Model
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import TextStreamer

from ai.dive.models.bitnet.bitnet import BitnetForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stop_token, tokenizer):
        super().__init__()
        stop_token_ids = tokenizer(stop_token, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze().tolist()
        
        self.stop_token_ids = stop_token_ids[1:] # remote the <s> token
        self.tokenizer = tokenizer
        self.last_n_tokens = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1].item()
        # print(f"Stops {self.stop_token_ids}")

        # add last token to last_n_tokens, and limit to len(stop_token_ids) - 1
        if len(self.last_n_tokens) >= len(self.stop_token_ids):
            self.last_n_tokens.pop(0)
        self.last_n_tokens.append(last_token)
        
        # print(f"Last tokens: {self.last_n_tokens}")

        # check if they are equal
        if self.last_n_tokens == self.stop_token_ids:
            return True

        return False

class StoppingTokenCriteria(StoppingCriteriaList):
    def __init__(self, stop_token, tokenizer):
        self.stop_token = stop_token
        self.tokenizer = tokenizer
        stopping_criteria = StoppingCriteriaSub(tokenizer=self.tokenizer, stop_token=stop_token)
        super().__init__([stopping_criteria])

class BitNetLLM(Model):
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__()
        
    def _build(self):
        print(f"Loading model {self.model_name}...")
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = BitnetForCausalLM.from_pretrained(self.model_name).to("cuda")

    # Function to run the model on a single example
    def _predict(self, data):
        prompt = data["prompt"]

        # Stop token
        stopping_criteria = StoppingTokenCriteria(stop_token="🐂", tokenizer=self.tokenizer)

        # Tokenize the data
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # Stream the results to the terminal so we can see it generating
        streamer = TextStreamer(self.tokenizer)

        generated_ids = self.model.generate(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=50,
            stopping_criteria=stopping_criteria
        )

        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = decoded[0][:-1]
        answer = answer.replace(prompt, "").strip()
        
        is_correct = answer.lower() in [d.lower() for d in data["answers"]]
        if data['answers'] == []:
            if "not in context" in answer.lower():
                is_correct = True

        return {
            "prompt": prompt,
            "guess": answer,
            "is_correct": is_correct,
            "model": self.model_name
        }
