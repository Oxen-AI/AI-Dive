from ai.dive.models.model import Model
from openai import OpenAI
import os

class TogetherAI(Model):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    #get api key stuff
    def _build(self):
        # If not api key, raise error
        if 'TOGETHER_API_KEY' not in os.environ:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")

        self.api_key = os.environ['TOGETHER_API_KEY']

    #function to run model on single prompt
    def _predict(self, data):
        #create openai client
        client = OpenAI(api_key=self.api_key,
                        base_url='https://api.together.xyz/v1',)
        messages = [
            {
                "role": "user",
                "content": f"{data['prompt']}"
            }
        ]

        # Make API call to Together API
        chat_completion = client.chat.completions.create(
            messages = messages,
            model= self.model_name,
            max_tokens=1024
        )
        response = chat_completion.choices[0].message.content
        print("-----------")
        print(data['prompt'])
        print(response)
        print("-----------")

        return {
            "model": self.model_name,
            "prompt": data['prompt'],
            "response": response,
        }