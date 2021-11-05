import argparse

from typing import Text

from torch.utils.data import Dataset

from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling

from transformers import Trainer, TrainingArguments

import torch
import jsonlines


class JsonlDataset(Dataset):

    def __init__(self,
                 dataset_path: Text,
                 tokenizer):
        super().__init__()

        with jsonlines.open(dataset_path) as reader:
            self.data = ['\n'.join(item['window']) for item in reader]

        batch_encoding = tokenizer(self.data,
                                   add_special_tokens=True,
                                   truncation=True,
                                   max_length=256)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __getitem__(self, idx: int):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


parser = argparse.ArgumentParser(description="train masked language model")

parser.add_argument(
    "--dataset",
    type=str,
    metavar="DATASET",
    help='path to dataset in jsonl format'
)

parser.add_argument(
    "--model",
    type=str,
    default='roBERTa',
    choices=['BERT', 'roBERTa'],
    help="transformer model to use (default:'roBERTa')"
)

parser.add_argument(
    "--output-dir",
    type=str,
    default='./mlm_output',
    help="directory in which to save model checkpoints (default:'./mlm_output')"
)

if __name__ == "__main__":

    args = parser.parse_args()

    if args.model == 'roBERTa':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
    else:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base')
        model = BertForMaskedLM.from_pretrained('bert-base')

    dataset = JsonlDataset(args.dataset, tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=25,
        per_gpu_eval_batch_size=3,
        save_steps=500,
        save_total_limit=2,
        seed=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )
    trainer.train()
    trainer.save_model('./bert_poetic_mlm')
