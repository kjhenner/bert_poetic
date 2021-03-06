import argparse
import json
import jsonlines
import os
from argparse import Namespace
from functools import partial

import torch
import torch.nn.functional as F

import tqdm

from models.poetry_classifier import PoetryClassifier
from helpers.classifier_inference_helper import predict_batch, preprocess, load_model
from helpers.utils import (
    gd_metadata_iter, 
    iter_lines_from_gd_path, 
    load_metadata,
    batch_iterable,
    iter_examples,
    iter_line_windows_from_gd_path
)


def parse_args():
    parser = argparse.ArgumentParser(description='Classify a file of example lines as poetry or not.')

    parser.add_argument('archive_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--vocab-path', type=str, default='pretrained')
    parser.add_argument('--batch-size', type=int, default=512)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_args = {
        'vocab_path': args.vocab_path
    }

    model = load_model(args.model_path)
    predict = partial(predict_batch, model)
    preprocess = partial(preprocess, model.tokenizer)

    metadata_list = load_metadata(args.archive_path,
                                  'gutenberg-dammit-files/gutenberg-metadata.json')
    poetry_metadata = list(gd_metadata_iter(metadata_list))

    pbar = tqdm.tqdm(total=len(poetry_metadata))
    with open(args.output_path, 'w') as f:
        for metadata_entry in poetry_metadata:
            pbar.set_description(f"reading {metadata_entry['Title'][0]}")
            pbar.update(1)
            ex_iter = iter_line_windows_from_gd_path(
                args.archive_path, 
                metadata_entry.get('gd-path')
            )
            batch_iter = batch_iterable(ex_iter, args.batch_size)
            out = ''
            for i, batch in enumerate(batch_iter):
                if len(batch) > 1:
                    for j, (ex, pred) in enumerate(zip(batch, predict(preprocess(batch)))):
                        line = i * args.batch_size + j
                        out += f"{str(int(pred >= 0.5))}\t{metadata_entry['gd-num-padded']}\t{line}\n"
            f.write(out)
