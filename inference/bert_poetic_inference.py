import torch
import random
from time import sleep
import argparse
from transformers import RobertaTokenizer, RobertaForMaskedLM


def parse_args():
    parser = argparse.ArgumentParser(description='Generate some text from a BERT MLM model')

    parser.add_argument('--model-path', type=str)
    parser.add_argument('--seed-text', type=str, nargs='+')

    return parser.parse_args()


def random_replace(model, tokenizer, text, spiciness=10):
    tokenized = tokenizer(text, return_tensors='pt')
    mask_idx = random.randrange(1, tokenized.input_ids.shape[-1])
    tokenized.input_ids[0][mask_idx] = tokenizer.mask_token_id
    tokens = tokenized['input_ids'].squeeze().tolist()
    output = model.forward(tokenized['input_ids'])

    rand_idx = int(random.randrange(0, spiciness))
    pred_token_id = torch.topk(output.logits[0][mask_idx], spiciness+1).indices[rand_idx]
    tokens[mask_idx] = pred_token_id

    return tokenizer.decode(tokens[1:-1])


def random_insert(model, tokenizer, text, spiciness=10):
    tokenized = tokenizer(text, return_tensors='pt')
    mask_idx = random.randrange(2, tokenized.input_ids.shape[-1] - 1)
    tokenized['input_ids'] = torch.cat((tokenized['input_ids'][0][:mask_idx],
                                        torch.tensor([tokenizer.mask_token_id]),
                                        tokenized['input_ids'][0][mask_idx:])).unsqueeze(0)
    tokenized['attention_mask'] = torch.cat((tokenized['attention_mask'][0][:mask_idx],
                                             torch.tensor([1]),
                                             tokenized['attention_mask'][0][mask_idx:])).unsqueeze(0)
    tokens = tokenized['input_ids'].squeeze().tolist()
    output = model.forward(tokenized['input_ids'])

    rand_idx = int(random.randrange(0, spiciness))
    pred_token_id = torch.topk(output.logits[0][mask_idx], spiciness+1).indices[rand_idx]
    tokens[mask_idx] = pred_token_id

    return tokenizer.decode(tokens[1:-1])


if __name__ == "__main__":
    args = parse_args()
    model = RobertaForMaskedLM.from_pretrained(args.model_path)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model.eval()
    spiciness = 14

    text = ' '.join(args.seed_text)
    prev_text = ''
    while True:
        sleep(.4)
        prev_text = text
        text = random_replace(model, tokenizer, text, spiciness=spiciness)
        if(text != prev_text):
            print(text)
            print('\n'*5)

        sleep(.4)
        prev_text = text
        text = random_replace(model, tokenizer, text, spiciness=spiciness)
        if(text != prev_text):
            print(text)
            print('\n'*5)

        sleep(.4)
        prev_text = text
        text = random_replace(model, tokenizer, text, spiciness=spiciness)
        if(text != prev_text):
            print(text)
            print('\n'*5)

        sleep(.4)
        prev_text = text
        text = random_replace(model, tokenizer, text, spiciness=spiciness)
        if(text != prev_text):
            print(text)
            print('\n'*5)
