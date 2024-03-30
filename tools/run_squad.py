
from ai.dive.saver import Saver
from ai.dive.diver import Diver
from ai.dive.data.prompt_template_filler import PromptTemplateFiller
from ai.dive.data.mmlu_dataset import MMLUDataset
from ai.dive.models.togetherai import TogetherAI
from ai.dive.models.anthropic import Anthropic
from ai.dive.models.unify import UnifyAI
import argparse

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run TogetherAI model on dataset')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='dataset to run model on')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    parser.add_argument('-n', '--num_samples', default=10, type=int, help='Number of samples to run model on')
    parser.add_argument('-s', '--save_every', default=10, type=int, help='How often to save results to file')
    args = parser.parse_args()

    # model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    # model = TogetherAI(model_name)
    # model = Anthropic("claude-3-opus-20240229")
    # model = Anthropic("claude-3-sonnet-20240229")
    # model = Anthropic("claude-instant-1.2")
    model = UnifyAI("mistral-medium@mistral-ai")
    # model = UnifyAI("gemma-2b-it@together-ai")
    dataset = MMLUDataset(args.dataset)

    output_keys = [
        'prompt',
        'choices',
        'answer',
        'response',
        'model',
        'time',
        'completion_tokens',
        'prompt_tokens',
        'total_tokens'
    ]
    saver = Saver(args.output, output_keys=output_keys, format="jsonl", save_every=args.save_every)
    diver = Diver(model, dataset, saver=saver)
    diver.run()


if __name__ == '__main__':
    main()