import argparse
import jsonlines
from helpers.utils import (
    iter_line_windows_from_gd_path_and_annotation_dict,
    load_annotation_dict,
    load_metadata,
    gd_metadata_iter
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate windowed examples for poetry classifier model.')
    parser.add_argument('archive_path', type=str)
    parser.add_argument('annotation_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--window-size', type=int, default=4)

    return parser.parse_args()


def main(args):
    annotation_dict = load_annotation_dict(args.annotation_path)
    gd_metadata_list = load_metadata(
        args.archive_path,
        'gutenberg-dammit-files/gutenberg-metadata.json'
    )
    poetry_metadata = gd_metadata_iter(gd_metadata_list)
    with jsonlines.open(args.output_path, 'w') as writer:
        for metadata_entry in poetry_metadata:
            data = iter_line_windows_from_gd_path_and_annotation_dict(
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
