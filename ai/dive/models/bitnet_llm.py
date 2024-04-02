
from ai.dive.models.model import Model
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import TextStreamer

from ai.dive.models.bitnet.bitnet import BitnetForCausalLM

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
        stop_token = "<STOP>"
        eos_token_id = self.tokenizer.encode(stop_token, add_special_tokens=False)[0]

        # Tokenize the data
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # Stream the results to the terminal so we can see it generating
        streamer = TextStreamer(self.tokenizer)

        generated_ids = self.model.generate(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=50
        )

        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = decoded[0]
        
        if stop_token in answer:
            answer = answer.split(stop_token)[-2].strip()
            answer = answer.split("\n")[-1].strip()

        return {
            "prompt": prompt,
            "extracted_answer": answer,
            "model": self.model_name
        }
