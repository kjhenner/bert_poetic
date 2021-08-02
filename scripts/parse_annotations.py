import json
import jsonlines
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse annotated poetry classifier examples."
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

    data = []

    with open(args.input_path) as f:
        lines = f.readlines()

    prev_lines = ['','','']
    for line in lines:
        label, text, index = line.split('\t')[:3]
        data.append({'label': int(label), 'text': text, 'index': index, 'prev_lines': prev_lines.copy()})
        prev_lines.pop(0)
        prev_lines.append(text)

    with jsonlines.open(args.output_path, 'w') as writer:
        writer.write_all(data)


if __name__ == "__main__":
    main(parse_args())
