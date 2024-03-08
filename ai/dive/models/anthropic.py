from ai.dive.models.model import Model
import os
import anthropic

#TODO: pass in model name to constructor
class Anthropic(Model):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    #get api key stuff
    def _build(self):
        # If not api key, raise error
        if 'ANTHROPIC_API_KEY' not in os.environ:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        self.api_key = os.environ['ANTHROPIC_API_KEY']

    #function to run model on single prompt
    def _predict(self, data):
        client = anthropic.Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=self.api_key,
        )
        prompt = data['prompt']
        message = client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            temperature=0,
            messages=[
                {
                    "role": "user",
                "content": f"{prompt}"
                }
            ]
        )
        print(message.content)
        response = message.content[0].text

        print("-----------")
        print(response)
        print("-----------")

        return {
            "prompt": prompt,
            "response": response,
        }