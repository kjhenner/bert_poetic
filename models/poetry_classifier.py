import argparse
from functools import partial
from typing import Dict

import torch
import torchmetrics
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.core.lightning import LightningModule

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

from datasets.jsonl_dataset import JsonlDataset


def preprocess(tokenizer, ex: Dict, max_length: int = 256) -> Dict:

    text = " [SEP] ".join(ex['window'])

    tokenized = tokenizer(
        text=text,
        add_special_tokens=True,
        max_length=max_length,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )

    if ex.get('label') is not None:
        tokenized['labels'] = torch.tensor([int(ex['label'])], dtype=torch.long)

    return tokenized


class PoetryClassifier(LightningModule):

    def __init__(self, hparams):
        super(PoetryClassifier, self).__init__()

        self.hparams = hparams

        if self.hparams.vocab_path == 'pretrained':
            self.transformer = RobertaForSequenceClassification.from_pretrained('roberta-base')
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        else:
            config = RobertaConfig(vocab_size=1000)
            self.transformer = RobertaForSequenceClassification(config=config)
            self.tokenizer = RobertaTokenizer(self.hparams.vocab_path, do_lower_case=True)

        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(),
            torchmetrics.Precision(num_classes=2, average='macro'),
            torchmetrics.Recall(num_classes=2, average='macro'),
            torchmetrics.F1(num_classes=2, average='macro')
        ])

        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, inputs):
        for k, v in inputs.items():
            inputs[k] = v.squeeze()
        return self.transformer(**inputs)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        self.log('loss', loss) 
        output = self.train_metrics(output.logits, batch['labels'])
        self.log_dict(output)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        self.log('val_loss', loss, on_epoch=True)
        output = self.val_metrics(output.logits, batch['labels'])
        self.log_dict(output)
        return loss

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        self.log('test_loss', loss, on_epoch=True)
        output = self.test_metrics(output.logits, batch['labels'])
        self.log_dict(output)
        return loss

    def train_dataloader(self):
        dataset = JsonlDataset(self.hparams.train_data,
                               preprocess=partial(preprocess, self.tokenizer))
        train_dataloader = DataLoader(dataset,
                                      batch_size=self.hparams.batch_size,
                                      num_workers=self.hparams.dataloader_workers,
                                      shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        dataset = JsonlDataset(self.hparams.val_data,
                               preprocess=partial(preprocess, self.tokenizer))
        val_dataloader = DataLoader(dataset,
                                    batch_size=self.hparams.batch_size,
                                    num_workers=self.hparams.dataloader_workers)
        return val_dataloader

    def test_dataloader(self):
        dataset = JsonlDataset(self.hparams.test_data,
                               preprocess=partial(preprocess, self.tokenizer))
        test_dataloader = DataLoader(dataset,
                                     batch_size=self.hparams.batch_size,
                                     num_workers=self.hparams.dataloader_workers)
        return test_dataloader

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0}
        ]
        opt = AdamW(optimizer_grouped_parameters, lr=lr, betas=(b1, b2))
        return opt


def parse_args():
    parser = argparse.ArgumentParser(description='Line classifier training script.')
    parser.add_argument('--max-epochs', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=0.000003)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=0.15)
    parser.add_argument('--train-data', type=str)
    parser.add_argument('--val-data', type=str)
    parser.add_argument('--test-data', type=str)
    parser.add_argument('--dataloader-workers', type=int, default=5)
    parser.add_argument('--vocab-path', type=str, default='pretrained')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    model = PoetryClassifier(args)
    trainer = pl.Trainer(gpus=1,
                         max_epochs=args.max_epochs,
                         accumulate_grad_batches=2,
                         logger=tb_logger)
    trainer.fit(model)
    torch.save(model.state_dict(), './line_classifier.pt')
    trainer.test(model)
