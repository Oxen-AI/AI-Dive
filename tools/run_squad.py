
from ai.dive.saver import Saver
from ai.dive.diver import Diver
from ai.dive.data.squad_dataset import SQuADDataset
from ai.dive.models.togetherai import TogetherAI
import argparse

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run TogetherAI model on dataset')
    parser.add_argument('-m', '--model', required=True, type=str, help='model to run on dataset')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='dataset to run model on')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    parser.add_argument('-n', '--num_samples', default=10, type=int, help='Number of samples to run model on')
    parser.add_argument('-s', '--save_every', default=10, type=int, help='How often to save results to file')
    args = parser.parse_args()

    # model_name = "meta-llama/Llama-2-70b-hf"
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    # model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    model_name = args.model
    model = TogetherAI(model_name)
    dataset = SQuADDataset(args.dataset)

    output_keys = ['id', 'prompt', 'question', 'context', 'answers', 'response', 'time']
    saver = Saver(args.output, output_keys=output_keys, format="jsonl", save_every=args.save_every)
    diver = Diver(model, dataset, saver=saver, num_samples=args.num_samples)
    diver.run()


if __name__ == '__main__':
    main()