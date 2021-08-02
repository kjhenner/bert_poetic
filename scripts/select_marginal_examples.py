import json
import jsonlines
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter jsonlines dataset to select items with marginal predictions and output for easy annotation."
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to the input data.'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='Path where the output will be saved'
    )
    return parser.parse_args()


def main(args):
    with jsonlines.open(args.input_path) as reader:
        lines = list(reader)

    with open(args.output_path, 'w') as f:
        for i, line in enumerate(lines):
            if 0.2 < line['pred'] < 0.8:
                marker = f"<<<<<<<<<< {line['pred']}\t{i}"
            else:
                marker = ""
            f.write(f"{int(line['pred'] > 0.5)}\t{line['line']}\t{i}\t{marker}\n")


if __name__ == "__main__":
    main(parse_args())
