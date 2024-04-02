
class QAPrompt:
    def __init__(self, example, n_shot_examples=[], should_add_answer=False):
        self.example = example
        self.n_shot_examples = n_shot_examples
        self.should_add_answer = should_add_answer

    def render(self):
        example = self.example
        system_msg = f"""The following is the context and question from a SQuAD dataset. Answer the question given the context. The answer should be a span of text from the context. If the answer is not in the context, write "Not in context."""
        
        if len(self.n_shot_examples) > 0:
            system_msg = ""
            for e in self.n_shot_examples:
                system_msg += f"Context:\n{e['context']}\n\nQuestion:\n{e['prompt']}\n\n"
                if e['answers'] == []:
                    system_msg += "Answer:\nNot in context.\n"
                else:
                    system_msg += f"Answer:\n{e['answers'][0]}\n"
                system_msg += "<STOP>\n\n"
            system_msg += "\n"

        prompt = f"""
{system_msg}

Context:
{example['context']}

Question:
{example['prompt']}

Answer:
"""
        if self.should_add_answer and example['answers'] != []:
            prompt += f"{example['answers'][0]}"
        elif self.should_add_answer:
            prompt += "Not in context."

        if self.should_add_answer:
            prompt += "\n<STOP>"

        return prompt
