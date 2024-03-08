from langchain.llms import CTransformers
from langchain.callbacks.base import BaseCallbackHandler
import sys

# Handler that prints each new token as it is computed
class NewTokenHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"{token}", end="", flush=True)

# Local CTransformers wrapper for Llama-2-7B-Chat
llm = CTransformers(
    model=sys.argv[1], # Location of downloaded GGML model
    model_type="llama", # Model type Llama
    stream=True,
    callbacks=[NewTokenHandler()],
    config={'max_new_tokens': 256, 'temperature': 0.01}
)

# Accept user input
while True:
    prompt = input('> ')
    output = llm(prompt)
    print(output)
