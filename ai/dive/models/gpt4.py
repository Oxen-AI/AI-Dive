
from transformers import ViTForImageClassification, ViTImageProcessor
import os

from ai.dive.models.model import Model
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

class GPT4(Model):
    def __init__(self):
        super().__init__()

    # Load model into memory
    def _build(self):
        # If not api key, raise error
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.api_key = os.environ['OPENAI_API_KEY']

    # Function to run the model on a single prompt
    def _predict(self, data):
        # Get the prompt from the data
        prompt = data['prompt']

        prompt = ChatPromptTemplate.from_template(prompt)
        model = ChatOpenAI(cache=False, temperature=0.9, model="gpt-4-1106-preview")
        output_parser = StrOutputParser()

        chain = prompt | model | output_parser
        response = chain.invoke({"input": ""})
        
        print("----------")
        print(response)
        print("-----------")

        return {
            "prompt": prompt,
            "response": response
        }
