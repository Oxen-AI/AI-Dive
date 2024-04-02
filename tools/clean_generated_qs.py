
from ai.dive.saver import Saver
from ai.dive.diver import Diver
from ai.dive.data.arc_dataset import ARCDataset
from ai.dive.models.togetherai import TogetherAI
import argparse
import json

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Clean up a generated question dataset')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='dataset to run model on')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    args = parser.parse_args()



if __name__ == '__main__':
    main()