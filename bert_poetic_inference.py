import torch
import random
from time import sleep
from transformers import BertTokenizer
from bert_poetic_model import BERTPoeticModel
import argparse
from argparse import Namespace


def parse_args():
    parser = argparse.ArgumentParser(description='Generate some text from a BERT MLM model')

    parser.add_argument('--model-path', type=str)
    parser.add_argument('--vocab-path', type=str)
    parser.add_argument('--seed-text', type=str, nargs='+')

    return parser.parse_args()


def random_replace(model, text):
    tokenized = model.tokenizer(text, return_tensors='pt')
    mask_idx = random.randrange(1, tokenized.input_ids.shape[-1])
    tokenized.input_ids[0][mask_idx] = model.tokenizer.mask_token_id
    tokens = tokenized['input_ids'].squeeze().tolist()
    output = model.forward(tokenized)

    pred_token_id = int(torch.argmax(output.logits[0][mask_idx]))
    tokens[mask_idx] = pred_token_id

    return model.tokenizer.decode(tokens[1:-1])


def random_insert(model, text):
    tokenized = model.tokenizer(text, return_tensors='pt')
    mask_idx = random.randrange(2, tokenized.input_ids.shape[-1] - 1)
    tokenized['input_ids'] = torch.cat((tokenized['input_ids'][0][:mask_idx],
                                        torch.tensor([model.tokenizer.mask_token_id]),
                                        tokenized['input_ids'][0][mask_idx:])).unsqueeze(0)
    tokenized['token_type_ids'] = torch.cat((tokenized['token_type_ids'][0][:mask_idx],
                                             torch.tensor([0]),
                                             tokenized['token_type_ids'][0][mask_idx:])).unsqueeze(0)
    tokenized['attention_mask'] = torch.cat((tokenized['attention_mask'][0][:mask_idx],
                                             torch.tensor([1]),
                                             tokenized['attention_mask'][0][mask_idx:])).unsqueeze(0)
    tokens = tokenized['input_ids'].squeeze().tolist()
    output = model.forward(tokenized)

    pred_token_id = int(torch.argmax(output.logits[0][mask_idx]))
    tokens[mask_idx] = pred_token_id

    return model.tokenizer.decode(tokens[1:-1])


if __name__ == "__main__":
    args = parse_args()
    model_args = {
        'vocab_path': args.vocab_path
    }
    #model_args = {
    #    'vocab_path': '/home/kevin/src/bert_poetic/bert-wordpiece-vocab.txt'
    #}
    model = BERTPoeticModel(Namespace(**model_args))

    #model.load_state_dict(torch.load('/mnt/atlas/models/model_2.pt'))
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    text = ' '.join(args.seed_text)
    print(text)
    prev_text = ''
    while True:
        sleep(.1)
        prev_text = text
        text = random_replace(model, text)
        if(text != prev_text):
            print(text)

        sleep(.1)
        prev_text = text
        text = random_replace(model, text)
        if(text != prev_text):
            print(text)
        
        sleep(.1)
        prev_text = text
        text = random_replace(model, text)
        if(text != prev_text):
            print(text)

        sleep(.1)
        prev_text = text
        text = random_insert(model, text)
        if(text != prev_text):
            print(text)
