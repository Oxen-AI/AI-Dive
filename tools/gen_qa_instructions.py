
from ai.dive.prompts.qa_prompt import QAPrompt
import argparse
import json

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Generate QA instructions for a dataset')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='dataset to run model on')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    args = parser.parse_args()

    qa_instructions = []
    with open(args.dataset, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            prompt = f"""
Context:
{data['context']}

Question:
{data['prompt']}

Answer:
"""
            answer = data['answers'][0] if len(data['answers']) > 0 else "Not in context."
            
            qa_instructions.append({
                'prompt': prompt.strip(),
                'response': answer,
            })
    
    with open(args.output, 'w') as f:
        for qa_instruction in qa_instructions:
            f.write(json.dumps(qa_instruction) + '\n')

if __name__ == '__main__':
    main()