import json
import argparse
import zipfile
import re

from helpers.utils import filtered_iter, zips_to_lines
from helpers.heuristic_classifier import matches_aparrish_poetry_heuristics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a labeled poetry dataset based on Allison Parish's heuristic filter."
    )
    parser.add_argument(
        'archive_path',
        type=str,
        help='Path to the GutenbergDammit archive.'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='Path where the output will be saved'
    )
    return parser.parse_args()


def main(args):
    zip_file = zipfile.ZipFile(args.archive_path)
    with zip_file.open('gutenberg-dammit-files/gutenberg-metadata.json') as f:
        gutenberg_metadata = json.load(f)
    filter_field = "Subject"
    pattern = r'^Poet.*'
    filtered_paths = filtered_iter(gutenberg_metadata, filter_field, pattern)
    items = zips_to_lines(zip_file, filtered_paths, include_metadata=True)
    with open(args.output_path, 'w') as f:
        prev_line = ''
        for line, metadata in items:
            matches = matches_aparrish_poetry_heuristics(prev_line, line)
            label = int(not any(matches))
            f.write(json.dumps({'line': line,
                                'prev_line': prev_line,
                                'label': label,
                                'gd-num': metadata['Num'],
                                'matches': matches}) + '\n')
            prev_line = line


if __name__ == "__main__":
    main(parse_args())
