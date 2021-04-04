import argparse
import glob
import json
import os
import zipfile
import tempfile
import re

from tokenizers import BertWordPieceTokenizer

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--archive_path",
        default=None,
        metavar="path",
        type=str,
        required=True,
        help="The zip file path to use as training"
    )

    parser.add_argument(
        "--out",
        default="./",
        type=str,
        help="Path to the output directory, where the files will be saved",
    )

    parser.add_argument(
        "--name", default="bert-wordpiece", type=str, help="The name of the output vocab files"
    )

    return parser.parse_args()


def zips_to_strings(zip_file, path_name_iter):
    for path_name in path_name_iter:
        with zip_file.open(os.path.join('gutenberg-dammit-files/', path_name)) as f:
            yield f.read().decode('utf-8')


def train_tokenizer(file_iterator):

    # Initialize an empty tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=True,
        lowercase=True,
    )

    # And then train
    #tokenizer.train_from_iterator(
    tokenizer.train_from_iterator(
        file_iterator,
        vocab_size=1000,
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=1000,
        wordpieces_prefix="##",
    )

    # Save the files
    tokenizer.save_model(args.out, args.name)


def filtered_iter(data, field, pattern):
    for item in data:
        matches_pattern = any([re.match(pattern, subj) for subj in item.get(field, [])])
        is_english = "English" in item.get("Language")
        if matches_pattern and is_english:
            yield item.get('gd-path')


def main(args):
    zip_file = zipfile.ZipFile(args.archive_path)

    with zip_file.open('gutenberg-dammit-files/gutenberg-metadata.json') as f:
        data = json.load(f)
    
    string_iter = zips_to_strings(zip_file, filtered_iter(data, "Subject", r'^Poet.*'))
    train_tokenizer(string_iter)


if __name__ == "__main__":
    args = parse_args()
    main(args)
