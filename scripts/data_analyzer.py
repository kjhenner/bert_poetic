import json
import jsonlines
import argparse
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze MLM training data to identify outliers."
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to the input data.'
    )
    return parser.parse_args()


def main(args):
    lengths = []
    with jsonlines.open(args.input_path) as reader:
        for item in reader:
            lengths.append(len(''.join(item['window'])))
            if len(''.join(item['window'])) > 650:
                print(item)

    data = np.histogram(lengths, bins=30)
    plt.title("Data length histogram")
    plt.hist(data, density=True, bins=30)
    plt.show()


if __name__ == "__main__":
    main(parse_args())
