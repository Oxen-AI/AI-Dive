from ai.dive.saver import Saver
from ai.dive.diver import Diver
from ai.dive.data.prompt_template_filler import PromptTemplateFiller
from ai.dive.models.gpt4 import GPT4
import argparse

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Complete a bunch of cheesy dad jokes')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='dataset to run model on')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    parser.add_argument('-n', '--num_samples', default=10, type=int, help='Number of samples to run model on')
    args = parser.parse_args()

    model = GPT4()

    # TODO: Show a few different prompts.
    # 1) You are a cheesy dad joke generator. Generate a punchline given a setup.\n\nSetup: {prompt}\n\nPunchline:
    # 2) You are a hilarious cheesy dad joke generator. Generate a creative afunny punchline given a setup.\n\nSetup: {prompt}\n\nPunchline: The setup should be unique and never heard before.
    # 3) You are a hilarious cheesy dad joke generator like Mitch Hedberg. Generate a random setup without a punchline for a joke without a punchline. The setup should be unique and never heard before. The setup should be funny and make sense. 

    dataset = PromptTemplateFiller(
        file=args.dataset,
        template="You are a hilarious cheesy dad joke generator. Generate a punchline given a setup.\n\nSetup: {prompt}\n\nPunchline:"
    )

    output_keys = ['idx', 'prompt', 'input', 'response', 'time']
    saver = Saver(args.output, output_keys=output_keys, format="csv", save_every=10)
    diver = Diver(model, dataset, saver=saver)
    diver.run()


if __name__ == '__main__':
    main()