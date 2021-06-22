import random
import argparse
from functools import partial
from typing import Dict

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.core.lightning import LightningModule

from transformers import BertConfig, BertForMaskedLM, BertTokenizer, AdamW

from datasets.filtered_gutenberg_dammit_dataset import FilteredGutenbergDammitDataset


def preprocess(tokenizer, ex: Dict, mask=True, max_length: int = 256) -> Dict:

    tokenized = tokenizer(
        text=ex["text"],
        add_special_tokens=True,
        max_length=max_length,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )
    if mask:
        mask_idx = random.randrange(1, torch.count_nonzero(tokenized.input_ids))
        tokenized['labels'] = torch.full((1,max_length),
                                         -100,
                                         dtype=torch.long)
        tokenized['labels'][0][mask_idx] = tokenized['input_ids'][0][mask_idx]
        tokenized['input_ids'][0][mask_idx] = tokenizer.mask_token_id

    return tokenized


class BERTPoeticModel(LightningModule):

    def __init__(self, hparams):
        super(BERTPoeticModel, self).__init__()

        self.hparams = hparams

        bert_config = BertConfig(vocab_size=1000)
        self.transformer = BertForMaskedLM(config=bert_config)
        self.tokenizer = BertTokenizer(self.hparams.vocab_path, do_lower_case=True)

    def forward(self, inputs):
        #for k, v in inputs.items():
        #    inputs[k] = v.squeeze()
        return self.transformer(**inputs)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        self.log('val_loss', loss)
        return loss

    def train_dataloader(self):
        dataset = FilteredGutenbergDammitDataset(self.hparams.train_data,
                                                 preprocess=partial(preprocess, self.tokenizer))
        train_dataloader = DataLoader(dataset,
                                      batch_size=self.hparams.batch_size,
                                      num_workers=self.hparams.dataloader_workers,
                                      shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        dataset = FilteredGutenbergDammitDataset(self.hparams.val_data,
                                                 preprocess=partial(preprocess, self.tokenizer))
        val_dataloader = DataLoader(dataset,
                                    batch_size=self.hparams.batch_size,
                                    num_workers=self.hparams.dataloader_workers)
        return val_dataloader

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
    parser = argparse.ArgumentParser(description='bert poetic MLM pretraining.')
    parser.add_argument('--max_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=0.000007)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0.15)
    parser.add_argument('--train_data', type=str, default='/mnt/atlas/gutenberg_dammit/gutenberg-dammit-files-v002.zip')
    parser.add_argument('--val_data', type=str, default='/mnt/atlas/gutenberg_dammit/gutenberg-dammit-files-v002.zip')
    parser.add_argument('--dataloader_workers', type=int, default=5)
    parser.add_argument('--vocab_path', type=str, default='/home/kevin/src/bert_poetic/bert-wordpiece-vocab.txt')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    model = BERTPoeticModel(args)
    #trainer = pl.Trainer(gpus=1, max_epochs=args['max_epochs'], logger=tb_logger)
    trainer = pl.Trainer(gpus=2,
                         max_epochs=args.max_epochs,
                         distributed_backend='ddp',
                         accumulate_grad_batches=5,
                         logger=tb_logger)
    trainer.fit(model)
    torch.save(model.state_dict(), '/mnt/atlas/models/model_2.pt')
