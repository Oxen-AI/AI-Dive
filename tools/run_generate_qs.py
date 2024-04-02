import sys
from ai.dive.saver import Saver
from ai.dive.diver import Diver
from ai.dive.data.latex_dataset import LatexDataset
from ai.dive.models.togetherai import TogetherAI
import argparse

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run TogetherAI model on dataset')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='dataset to run model on')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    parser.add_argument('-n', '--num_samples', default=10, type=int, help='Number of samples to run model on')
    args = parser.parse_args()

    # model_name = "meta-llama/Llama-2-70b-hf"
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model = TogetherAI(model_name)
    dataset = LatexDataset(args.dataset)

    output_keys = ['file', 'title', 'abstract', 'content', 'prompt', 'response', 'time']
    saver = Saver(args.output, output_keys=output_keys, save_every=1)
    diver = Diver(model, dataset, saver=saver)
    diver.run()


if __name__ == '__main__':
    main()