import sys
sys.path.append('C:/Users/nyc8p/OneDrive/Documents/GitHub/AI-Dive/')
from ai.dive.saver import Saver
from ai.dive.diver import Diver
from ai.dive.data.prompt_template_filler import PromptTemplateFiller
from ai.dive.data.togetheraipromptrepeater import togetheraipromptrepeater
from ai.dive.models.togetherai import TogetherAI
import json
import argparse

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run TogetherAI model on dataset')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='dataset to run model on')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    parser.add_argument('-n', '--num_samples', default=10, type=int, help='Number of samples to run model on')
    args = parser.parse_args()

    model_name = "meta-llama/Llama-2-70b-hf"
    model = TogetherAI(model_name)

    #arc-easy dataset jsonl file from directory (arc_easy_test.jsonl)
    # Load dataset from JSONL file
    #prompt template
    dataset = togetheraipromptrepeater(
        file=args.dataset,
        template="You are an AI assistant, you will be given a question and 4 answer choices, output the correct answer choice with no other text"
    )
    # dataset = PromptTemplateFiller(
    #     file=args.dataset,
    #     template="You are an AI assistant, you will be given a question and 4 answer choices, output the correct answer choice with no other text"
    # )

    output_keys = ['id', 'prompt', 'choices', 'answer_idx', 'predicted_idx']
    saver = Saver(args.output, output_keys=output_keys, format="csv", save_every=10)
    diver = Diver(model, dataset, saver=saver)
    diver.run()


if __name__ == '__main__':
    main()