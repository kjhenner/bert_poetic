import json
import jsonlines
import argparse
import random
import zipfile

from helpers.utils import load_metadata, iter_lines_from_pd_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select representative portions for further annotation."
    )
    parser.add_argument(
        'archive_path',
        type=str,
        help='Path to poetry_dammit zip file.'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='Path where the output will be saved'
    )
    return parser.parse_args()


def gd_path_to_classified_path(gd_path):
    return (gd_path.split('.')[0] + '_line_classified.jsonl').replace('/', '.')


def main(args):

    metadata = load_metadata(args.archive_path, 'poetry-metadata.json')
    paths = [gd_path_to_classified_path(item['gd-path']) for item in metadata]

    with open(args.output_path, 'w') as f:
        for path in paths:
            lines = list(iter_lines_from_pd_path(args.archive_path, path))
            half = int(len(lines) / 2)
            for i, line in enumerate(lines):
                line['line_index'] = i
                line['path'] = path
            lines = lines[:100] + lines[half - 50:half + 50] + lines[-100:]
            for line in lines:
                if 0.2 < line['pred'] < 0.8:
                    marker = f"<<<<<<<<<< {line['pred']}\t{i}"
                else:
                    marker = ""
                f.write(f"{int(line['pred'] > 0.5)}\t{line['line']}\t{line['path']}.{line['line_index']}\t{marker}\n")


if __name__ == "__main__":
    main(parse_args())
