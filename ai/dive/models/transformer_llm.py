
from ai.dive.models.model import Model
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import TextStreamer

class TransformerLLM(Model):
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__()
        
    def _build(self):
        print(f"Loading model {self.model_name}...")
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = LlamaForCausalLM.from_pretrained(self.model_name)

    # Function to run the model on a single example
    def _predict(self, data):
        prompt = data["prompt"]

        # Tokenize the data
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # Stream the results to the terminal so we can see it generating
        streamer = TextStreamer(self.tokenizer)

        generated_ids = self.model.generate(
            **model_inputs,
            do_sample=True,
            streamer=streamer,
            num_return_sequences=1,
            max_new_tokens=128
        )

        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = decoded[0]

        return answer
