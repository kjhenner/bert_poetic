import sys
import random
import os
import jsonlines
from collections import defaultdict
from itertools import chain

def write_examples(examples, path):
    with open(path, 'w') as f:
        f.writelines(examples)

def train_dev_test_split(examples, train_prop, input_file):

    dirname, basename = os.path.split(input_file)

    example_count = len(examples)

    train_par = int(example_count * train_prop)
    test_par = int(example_count/2 * (1-train_prop))

    split_data = {
        'train': examples[:train_par],
        'validation': examples[train_par:-test_par],
        'test': examples[-test_par:]
    }

    for k, v in split_data.items():
        print(f"{k}: {len(v)} examples")
        path = os.path.join(dirname, f"{k}.{basename}")
        write_examples(v, path)
    
if __name__ == "__main__":
    input_file = sys.argv[1]
    split_prop = float(sys.argv[2])
    random.seed(42069)

    with open(input_file) as f:
        examples = f.readlines()
    random.shuffle(examples) 
    train_dev_test_split(examples, split_prop, input_file)
