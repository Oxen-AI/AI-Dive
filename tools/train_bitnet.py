
from ai.dive.saver import Saver
from ai.dive.trainers.bitnet_trainer import BitNetTrainer
# TODO: Combine PR and QA into one module
from ai.dive.data.sft.pr_sft_dataset import SFTDataModule
from ai.dive.models.bitnet_llm import BitNetLLM
import argparse
import os
import json

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train a BitNet')
    parser.add_argument('-m', '--model', required=True, type=str, help='base model to start with')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='dataset to train model on')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    parser.add_argument('-n', '--num_samples', default=-1, type=int, help='how many examples to train on')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size for training')
    parser.add_argument('--max_seq_len', default=512, type=int, help='max sequence length for model')
    args = parser.parse_args()

    # mkdir output if not exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model = BitNetLLM(args.model)
    
    # TODO: This can just take a prompt object
    dataset = SFTDataModule(
        tokenizer=model.tokenizer,
        data_path=args.dataset,
        num_samples=args.num_samples,
        max_seq_len=args.max_seq_len
    )
    print(model.tokenizer.decode(dataset.dataset[0]['input_ids']))
    
    # save some of the training data for debugging
    with open(os.path.join(args.output, "debug_data.jsonl"), "w") as f:
        for i in range(5):
            data = {
                "text": model.tokenizer.decode(dataset.dataset[i]['input_ids']),
            }
            f.write(json.dumps(data) + "\n")

    trainer = BitNetTrainer(args.output, batch_size=args.batch_size)
    trainer.train(model, dataset)

if __name__ == '__main__':
    main()