from ai.dive.models.model import Model
from openai import OpenAI
import os

#PLAN
#TODO: finish the predict() function
#TODO: finish run_togetherai.py

#TODO: pass in model name to constructor
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
        #get prompt from data
        prompt = data['prompt']
        messages = [
            {
                "role": "user",
                "content": f"{prompt}"
            }
        ]
        
        # Make API call to Together API
        chat_completion = client.chat.completions.create(
            messages = messages,
            model= self.model_name,
            max_tokens=1024
        )
        response = chat_completion.choices[0].message.content
        # lines = response.split("\n")
        # response = lines[1].strip() #get the response from the second line
        print("-----------")
        print(response)
        print("-----------")

        return {
            "prompt": prompt,
            "response": response,
        }