import argparse
import json
from argparse import Namespace
from functools import partial
import jsonlines
import os

import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from models.poetry_classifier import PoetryClassifier

from datasets.jsonl_dataset import JsonlDataset

from helpers.utils import batch_iterable
from helpers.utils import gd_metadata_iter, iter_lines_from_gd_path, load_metadata


def parse_args():
    parser = argparse.ArgumentParser(description='Classify a file of example lines as poetry or not.')

    parser.add_argument('archive_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--model-path', type=str, default='/mnt/atlas/models/poetry_classifier_annotated.pt')
    parser.add_argument('--vocab-path', type=str, default='pretrained')
    parser.add_argument('--batch-size', type=int, default=512)

    return parser.parse_args()


def predict_batch(model, batch):
    with torch.no_grad():
        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}
        output = model.forward(batch)
        return F.softmax(output['logits'], dim=1)[:, 1].tolist()


def iter_examples(archive_path, metadata_entry):
    prev_lines = ['','','']
    for line in iter_lines_from_gd_path(archive_path, metadata_entry.get('gd-path')):
        text = "[SEP]".join(prev_lines) + " [SEP] " + line
        ex = {'text': text, 'line': line}
        prev_lines.pop(0)
        prev_lines.append(line)
        yield ex


def preprocess(tokenizer, input_batch, max_length=256):
    text = [ex['text'] for ex in input_batch]
    return tokenizer(
        text = text,
        max_length = max_length,
        return_tensors = 'pt',
        padding='max_length',
        truncation=True,
        is_split_into_words=False
    )


if __name__ == "__main__":
    args = parse_args()
    model_args = {
        'vocab_path': args.vocab_path
    }

    model = PoetryClassifier(Namespace(**model_args))
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda(device=0)
    predict = partial(predict_batch, model)
    preprocess = partial(preprocess, model.tokenizer)

    metadata_list = load_metadata(args.archive_path)
    poetry_metadata = list(gd_metadata_iter(metadata_list))
    poetry_metadata_path = os.path.join(args.output_dir, 'poetry-metadata.json')
    with open(poetry_metadata_path, 'w') as f:
        f.write(json.dumps(poetry_metadata, indent=2))
    pbar = tqdm.tqdm(total=len(poetry_metadata))
    for metadata_entry in poetry_metadata:
        pbar.set_description(f"reading {metadata_entry['Title'][0]}")
        pbar.update(1)
        metadata_entry['gd-poetry-path'] = metadata_entry['gd-path'].split('.')[0] + '.json'
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
                                title = negative_buffer[-8:]
                                negative_buffer = []
                            else:
                                poem = {
                                    'title': '\n'.join([line for line in title if line]),
                                    'text': poetry_buffer
                                }
                                poems.append(poem)
                                title = negative_buffer[-8:]
                                poetry_buffer = []
                                poetry_buffer.append(ex['line'])
                                negative_buffer = []
                    else:
                        negative_buffer.append(ex['line'])
        write_path = os.path.join(args.output_dir, metadata_entry['gd-poetry-path'])
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        with open(write_path, 'w') as f:
            f.write(json.dumps(poems, indent=2))

