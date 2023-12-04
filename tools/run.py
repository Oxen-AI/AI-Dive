from ai.dive.saver import Saver
from ai.dive.diver import Diver
from ai.dive.vit import Vit
from ai.dive.file_classification import FileClassification
import argparse

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run VIT model on dataset')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='dataset to run model on')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    parser.add_argument('-n', '--num-samples', default=-1, type=int, help='Number of samples to run model on')
    args = parser.parse_args()

    model = Vit()
    dataset = FileClassification(
        data_dir=args.dataset,
        file="test.csv",
        path_key="filename",
        label_key="class_name"
    )
    saver = Saver(args.output, format="csv", save_every=10)
    diver = Diver(model, dataset, saver=saver)
    results = diver.run(args)
    print(results)

    # Run the model on one example
    # output = model.process({
    #     'filename': args.dataset,
    #     'class_name': 'shiba inu'
    # })
    # print(output)
    
    # Run the model on a whole dataset
    
    
    # model.run({
    #     'dataset': args.dataset,
    #     'output': args.output,
    #     'num_samples': args.num_samples
    # })
    
if __name__ == '__main__':
    main()