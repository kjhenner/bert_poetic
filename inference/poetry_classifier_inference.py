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
    iter_examples
)


def parse_args():
    parser = argparse.ArgumentParser(description='Classify a file of example lines as poetry or not.')

    parser.add_argument('archive_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--model-path', type=str, default='/mnt/atlas/models/poetry_classifier_annotated.pt')
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

    os.makedirs(args.output_dir, exist_ok=True)
    metadata_list = load_metadata(args.archive_path,
                                  'gutenberg-dammit-files/gutenberg-metadata.json')
    poetry_metadata = list(gd_metadata_iter(metadata_list))
    poetry_metadata_path = os.path.join(args.output_dir, 'poetry-metadata.json')
    pbar = tqdm.tqdm(total=len(poetry_metadata))
    for metadata_entry in poetry_metadata:
        pbar.set_description(f"reading {metadata_entry['Title'][0]}")
        pbar.update(1)
        path = (metadata_entry['gd-path'].split('.')[0] + '_line_classified.jsonl').replace('/', '.')
        metadata_entry['gd-line-classified-path'] = path
        ex_iter = iter_examples(args.archive_path, metadata_entry)
        batch_iter = batch_iterable(ex_iter, args.batch_size)
        data = []
        for batch in batch_iter:
            if len(batch) > 1:
                for ex, pred in zip(batch, predict(preprocess(batch))):
                    ex['pred'] = pred
                    data.append(ex)
        write_path = os.path.join(args.output_dir, path)
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        with jsonlines.open(write_path, 'w') as writer:
            writer.write_all(data)
    with open(poetry_metadata_path, 'w') as f:
        f.write(json.dumps(poetry_metadata, indent=2))
