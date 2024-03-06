from ai.dive.models.model import Model
from openai import OpenAI
import together
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
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.api_key = os.environ['OPENAI_API_KEY']

    #function to run model on single prompt
    def _predict(self, data):
        #create openai client
        client = OpenAI(api_key=self.api_key,
                        base_url='https://api.together.xyz/v1',)
        #get prompt from data
        prompt = data['prompt']
        choices = data['choices']
         # Prepare messages for API call
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant, you will be given a question and 4 answer choices, output the correct answer choice with no other text",  # could change this later
            },
            {
                "role": "user",
                "content": f"{prompt}\n\n Please choose the best answer among these options: {', '.join(choices)}",  # Include choices in the prompt
            }
        ]
        
        # Make API call to Together API
        chat_completion = client.chat.completions.create(
            messages = messages,
            model= self.model_name,
            stop = "<|im_end|>",
            max_tokens=1024
        )
        response = chat_completion.choices[0].message.content
        lines = response.split("\n")
        response = lines[1].strip() #get the response from the second line
        #let response be one of the strings in choices if the string in choices is in response
        for choice in choices:
            if choice in response:
                selected_choice = choice
                print(selected_choice)
                break
        response = selected_choice 
        predictedidx = choices.index(response) #get index of response from choices
        answeridx = data['answer_idx']
        id = data['id']
        #include id

        return {
            "id": id,
            "prompt": prompt,
            'choices': choices,
            "predictedidx": predictedidx,
            "answeridx": answeridx
        }