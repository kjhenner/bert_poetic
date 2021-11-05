import argparse
import json
import jsonlines
import os
from argparse import Namespace

import tqdm

from helpers.utils import (
    gd_metadata_iter, 
    iter_lines_from_gd_path, 
    load_metadata,
    batch_iterable,
    iter_examples
)


def parse_args():
    parser = argparse.ArgumentParser(description='Classify a file of example lines as poetry or not.')

    parser.add_argument('archive_path', type=str)
    parser.add_argument('output_dir', type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    metadata_list = load_metadata(args.archive_path)
    poetry_metadata = list(gd_metadata_iter(metadata_list))

    with open(poetry_metadata_path, 'w') as f:
        f.write(json.dumps(poetry_metadata, indent=2))

    for metadata_entry in poetry_metadata:
        pbar.set_description(f"reading {metadata_entry['Title'][0]}")
        pbar.update(1)
        metadata_entry['gd-poetry-path'] = metadata_entry['gd-path'].split('.')[0] + '.json'
        metadata_entry['gd-poetry-path'] = metadata_entry['gd-poetry-path'].replace('/', '.')
        ex_iter = iter_examples(args.archive_path, metadata_entry)
        batch_iter = batch_iterable(ex_iter, args.batch_size)
        poems = []
        negative_buffer = ['','','']
        poetry_buffer = []
        title = []
        for batch in batch_iter:
            if len(batch) > 1:
                for ex, pred in zip(batch, predict(preprocess(batch))):
                    if pred > 0.5:
                        if len(negative_buffer) < 3:
                            poetry_buffer += negative_buffer
                            negative_buffer = []
                            poetry_buffer.append(ex['line'])
                        else:
                            if not poetry_buffer:
                                poetry_buffer.append(ex['line'])
                                negative_buffer = []
                            else:
                                poem = {
                                    'title': '\n'.join([line for line in title if line]),
                                    'text': poetry_buffer
                                }
                                poems.append(poem)
                                poetry_buffer = []
                                poetry_buffer.append(ex['line'])
                                negative_buffer = []
                    else:
                        negative_buffer.append(ex['line'])
        write_path = os.path.join(args.output_dir, metadata_entry['gd-poetry-path'])
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        with open(write_path, 'w') as f:
            f.write(json.dumps(poems, indent=2))

