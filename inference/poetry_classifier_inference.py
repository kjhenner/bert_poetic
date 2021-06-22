import argparse
import json
from argparse import Namespace
from functools import partial
import jsonlines

import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from models.poetry_classifier import PoetryClassifier
from models.poetry_classifier import preprocess
from datasets.jsonl_dataset import JsonlDataset
from helpers.utils import batch_iterable

def parse_args():
    parser = argparse.ArgumentParser(description='Classify a file of example lines as poetry or not.')

    parser.add_argument('data_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--model-path', type=str, default='/mnt/atlas/models/poetry_classifier.pt')
    parser.add_argument('--vocab-path', type=str, default='pretrained')
    parser.add_argument('--batch-size', type=int, default=512)

    return parser.parse_args()


def predict_batch(model, batch):
    with torch.no_grad():
        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}
        output = model.forward(batch)
        return F.softmax(output['logits'], dim=1)[:, 1].tolist()


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
    dataset = JsonlDataset(args.data_path,
                           preprocess=partial(preprocess, model.tokenizer),)
    data = DataLoader(dataset,
                      batch_size=args.batch_size,
                      num_workers=5,
                      pin_memory=True)

    progress = tqdm.tqdm(total=len(dataset))

    with jsonlines.open(args.data_path) as reader:
        line_iter = batch_iterable(list(reader), n=args.batch_size)

    with open(args.output_path, 'w') as f:
        for data_batch, input_batch in zip(iter(data), line_iter):
            for line, pred in zip(input_batch, predict(data_batch)):
                line['pred'] = pred
                f.write(json.dumps(line) + '\n')
            progress.update(args.batch_size)
