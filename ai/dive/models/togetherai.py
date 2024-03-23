import sys
sys.path.append('C:/Users/nyc8p/OneDrive/Documents/GitHub/AI-Dive/')
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
        #get prompt for model from data
        modelprompt = data['modelprompt']
        print(modelprompt)
        messages = [
            {
                "role": "user",
                "content": f"{modelprompt}"
            }
        ]
        
        # Make API call to Together API
        chat_completion = client.chat.completions.create(
            messages = messages,
            model= self.model_name,
            max_tokens=1024
        )
        response = chat_completion.choices[0].message.content
        #now make response be the string after 'Answer: '
        print(response)
        #if 'Answer: ' in response:
        if 'Answer: ' in response:
            response = response.split('Answer: ')[1]

        #save prompt to be string after \n\n, before second \n\n
        print("-----------")
        print(response)
        print("-----------")

        return {
            "model": self.model_name,
            "system_msg": data['system_msg'],
            "prompt": data['prompt'],
            "response": response,
        }