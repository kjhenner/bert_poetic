import argparse
import os

import jsonlines
from helpers.utils import (
    iter_line_windows_in_poetry_segments_from_gd_path_and_annotation_dict,
    load_annotation_dict_from_label_list,
    load_metadata,
    gd_metadata_iter
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate windowed data from contiguous poetry sequences for MLM.')
    parser.add_argument('archive_path', type=str)
    parser.add_argument('annotation_dir', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--window-size', type=int, default=4)

    return parser.parse_args()


def main(args):
    annotation_dict = {}
    for path in os.listdir(args.annotation_dir):
        annotation_dict.update(load_annotation_dict_from_label_list(os.path.join(args.annotation_dir, path)))

    gd_metadata_list = load_metadata(
        args.archive_path,
        'gutenberg-dammit-files/gutenberg-metadata.json'
    )
    poetry_metadata = gd_metadata_iter(gd_metadata_list)
    with jsonlines.open(args.output_path, 'w') as writer:
        for i, metadata_entry in enumerate(poetry_metadata):
            data = iter_line_windows_in_poetry_segments_from_gd_path_and_annotation_dict(
                args.archive_path,
                metadata_entry['gd-path'],
                metadata_entry['gd-num-padded'],
                annotation_dict,
                int(args.window_size)
            )
            writer.write_all(data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
