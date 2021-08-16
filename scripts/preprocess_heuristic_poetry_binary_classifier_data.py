import json
import argparse
import zipfile
import re

from helpers.utils import gd_metadata_iter, iter_lines_from_gd_path, load_metadata
from helpers.heuristic_classifier import matches_aparrish_poetry_heuristics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a labeled poetry dataset based on Allison Parrish's heuristic filter."
    )
    parser.add_argument(
        'archive-path',
        type=str,
        help='Path to the GutenbergDammit archive.'
    )
    parser.add_argument(
        'output-path',
        type=str,
        help='Path where the output will be saved'
    )
    return parser.parse_args()


def main(args):

    with open(args.output_path, 'w') as f:
        prev_line = ''
        metadata_list = load_metadata(args.archive_path)
        for metadata_entry in gd_metadata_iter(metadata_list):
            print(f"reading {metadata_entry['Title'][0]}")
            for line in iter_lines_from_gd_path(
                    args.archive_path,
                    metadata_entry.get('gd-path')):
                matches = matches_aparrish_poetry_heuristics(prev_line, line)
                label = int(not any(matches))
                f.write(json.dumps({'line': line,
                                    'prev_line': prev_line,
                                    'label': label,
                                    'gd-num': metadata_entry['Num'],
                                    'matches': matches}) + '\n')
                prev_line = line


if __name__ == "__main__":
    main(parse_args())
