from ai.dive.saver import Saver
from ai.dive.diver import Diver
from ai.dive.models.clip import CLIP
from ai.dive.data.file_classification import FileClassification
import argparse

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run model on dataset')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='dataset to run model on')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    parser.add_argument('-n', '--num-samples', default=-1, type=int, help='Number of samples to run model on')
    args = parser.parse_args()

    model = CLIP()
    dataset = FileClassification(
        data_dir=args.dataset,
        file="images.csv",
        path_key="filename",
        label_key="class_name",
    )

    output_keys = ['filename', 'class_name', 'prediction', 'probability', 'is_correct', 'time']
    saver = Saver(args.output, output_keys=output_keys, format="csv", save_every=10)

    diver = Diver(model, dataset, saver=saver)
    diver.run()

    # Run the model on one example
    # data = {"filepath": "images/shiba_inu_1.jpg"}
    # output = model.predict(data)
    # print(output)

if __name__ == '__main__':
    main()