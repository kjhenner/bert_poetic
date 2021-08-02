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


def load_model(model_path, vocab_path='pretrained'):
    model_args = {
        'vocab_path': vocab_path
    }

    model = PoetryClassifier(Namespace(**model_args))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda(device=0)

    return model


def predict_batch(model, batch):
    with torch.no_grad():
        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}
        output = model.forward(batch)
        return F.softmax(output['logits'], dim=1)[:, 1].tolist()


def predict_data(data_path, model, batch_size=512):
    predict = partial(predict_batch, model)
    dataset = JsonlDataset(data_path,
                           preprocess=partial(preprocess, model.tokenizer))
    data = DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=5,
                      pin_memory=True)

    progress = tqdm.tqdm(total=len(dataset))
    with jsonlines.open(data_path) as reader:
        line_iter = batch_iterable(list(reader), n=batch_size)

    preds = []
    for data_batch, input_batch in zip(iter(data), line_iter):
        preds += predict(data_batch)
        progress.update(batch_size)
    return preds
