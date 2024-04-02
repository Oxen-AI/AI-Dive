
from ai.dive.models.model import Model
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import TextStreamer

from ai.dive.models.bitnet.bitnet import BitnetForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops, tokenizer):
        super().__init__()
        self.stops = stops
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        # print(f"Stops {self.stops}")
        # print(f"Last token: {last_token} -> {self.tokenizer.decode(last_token)}")
        for stop in self.stops:
            if self.tokenizer.decode(stop) == self.tokenizer.decode(last_token):
                return True
        return False

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
        stop_words = ["<", "\n\n"]
        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer=self.tokenizer, stops=stop_words_ids)])

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
