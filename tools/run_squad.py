
from ai.dive.saver import Saver
from ai.dive.diver import Diver
from ai.dive.data.squad_dataset import SquadDataset
from ai.dive.models.bitnet_llm import BitNetLLM
from ai.dive.models.bitnet_olmo import BitNetOlmo
import argparse

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run TogetherAI model on dataset')
    parser.add_argument('-m', '--model', required=True, type=str, help='model to run on dataset')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='dataset to run model on')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    parser.add_argument('-n', '--num_samples', default=10, type=int, help='Number of samples to run model on')
    parser.add_argument('-s', '--save_every', default=1, type=int, help='How often to save results to file')
    parser.add_argument('--n_shot', type=str, help='n_shot to prepopulate prompt with')
    args = parser.parse_args()

    model = BitNetLLM(args.model)
    # model = BitNetOlmo(args.model)
    if args.n_shot:
        dataset = SquadDataset(args.dataset, n_shot_file=args.n_shot)
    else:
        dataset = SquadDataset(args.dataset)

    output_keys = [
        'prompt',
        'question',
        'context',
        'answers',
        'guess',
        'is_correct',
        'model',
        'time'
    ]
    saver = Saver(args.output, output_keys=output_keys, format="jsonl", save_every=args.save_every)
    diver = Diver(model, dataset, saver=saver, num_items=args.num_samples)
    diver.run()


if __name__ == '__main__':
    main()