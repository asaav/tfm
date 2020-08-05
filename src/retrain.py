from hnExtraction import example_extraction
from hnMining import continue_training
from comargs import retrain_args

import os
import re
import sys


def main():
    args = retrain_args()
    if not os.path.exists(args.model):
        sys.exit("Model " + args.model + " does not exist")
    if not os.path.exists(args.refs):
        sys.exit("Path " + args.refs + " directory does not exist")
    if not os.path.exists(args.data):
        sys.exit("Path " + args.data + " directory does not exist")

    it_regex = r'(it(\d+))(.+?)\.'
    match = re.match(it_regex, os.path.basename(args.model))
    it_number = int(match.group(2))
    name = match.group(3)

    model_path = args.model
    dataset_path = args.data
    if it_number > 0:
        dataset_path = os.path.join("hnmData", name, "it" + str(it_number-1))

    for i in range(args.iterations):
        example_extraction(model_path, dataset_path, args.refs)

        # example_extraction outputs a new folder with in hnmData/
        continue_training(model_path, "hnmData")

        it_number += 1  # increment it_number
        head, _ = os.path.split(model_path)
        model_path = os.path.join(head, "it" + str(it_number) + name + ".h5")


if __name__ == "__main__":
    main()
