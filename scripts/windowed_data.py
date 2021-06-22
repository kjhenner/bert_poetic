import zipfile
import json
import os
import jsonlines
import argparse
from itertools import islice
from helpers.utils import filtered_iter, windowed_iter



def parse_args():
    parser = argparse.ArgumentParser(description='Iterate over poetry data and yield windowed data')
    parser.add_argument('output_path', type=str)
    parser.add_argument('--archive-path', type=str)
    parser.add_argument('--window-size', type=int, default=3)
    parser.add_argument('--per-book-limit', type=int, default=0)

    return parser.parse_args()


def main(args):
    zip_file = zipfile.ZipFile(args.archive_path)
    with zip_file.open('gutenberg-dammit-files/gutenberg-metadata.json') as f:
        gutenberg_metadata = json.load(f)

    filtered_items = filtered_iter(gutenberg_metadata, "Subject", r'^Poet.*')

    data = []
    for item in filtered_items:
        with zip_file.open(os.path.join('gutenberg-dammit-files', item.get('gd-path'))) as f:
            lines = f.read().decode('utf-8').split('\n')
        w_iter = windowed_iter(lines, args.window_size)
        if args.per_book_limit:
            witer = islice(w_iter, args.per_book_limit)
        for window in w_iter: 
            line = window[args.window_size]
            display = list(window)
            display[args.window_size] = ">>>" + line
            data.append({"title": item.get('Title'),
                         "text": '\n'.join(display),
                         "window": window,
                         "gd-path": item.get('gd-path')})
    with jsonlines.open(args.output_path, 'w') as writer:
        writer.write_all(data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
