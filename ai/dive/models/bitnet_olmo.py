
from ai.dive.models.model import Model
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer

from transformers import TextStreamer
import torch

class BitNetOlmo(Model):
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__()
        
    def _build(self):
        print(f"Loading model {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

    # Function to run the model on a single example
    def _predict(self, data):
        prompt = data["prompt"]


        # Stream the results to the terminal so we can see it generating
        streamer = TextStreamer(self.tokenizer)

        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.8, repetition_penalty=1.1, do_sample=True,streamer=streamer)
        generated = pipe(prompt,  max_new_tokens=50)
        
        answer = generated[0]["generated_text"]
        answer = answer.replace(prompt, "").strip()
        print(answer)

        is_correct = answer.lower() in [d.lower() for d in data["answers"]]

        return {
            "prompt": prompt,
            "guess": answer,
            "is_correct": is_correct,
            "model": self.model_name
        }
