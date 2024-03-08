from ai.dive.models.model import Model
from openai import OpenAI
import os

class UnifyAI(Model):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def _build(self):
        # If not api key, raise error
        if 'UNIFY_API_KEY' not in os.environ:
            raise ValueError("UNIFY_API_KEY not found in environment variables")

        self.api_key = os.environ['UNIFY_API_KEY']

    def _predict(self, data):
        # create openai client
        client = OpenAI(api_key=self.api_key,
                        base_url='https://api.unify.ai/v0/',)
        # get prompt from data
        prompt = data['prompt']
        messages = [
            {
                "role": "user",
                "content": f"{prompt}"
            }
        ]
        
        # Make API call
        chat_completion = client.chat.completions.create(
            messages = messages,
            model= self.model_name,
            max_tokens=1024
        )
        response = chat_completion.choices[0].message.content
        # lines = response.split("\n")
        # response = lines[1].strip() #get the response from the second line
        print(chat_completion.usage)
        print("-----------")
        print(response)
        print("-----------")

        return {
            "prompt": prompt,
            "response": response,
            "model": self.model_name,
            "completion_tokens": chat_completion.usage.completion_tokens, 
            "prompt_tokens": chat_completion.usage.prompt_tokens,
            "total_tokens": chat_completion.usage.total_tokens
        }