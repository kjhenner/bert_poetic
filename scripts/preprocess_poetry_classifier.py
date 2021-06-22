import argparse
import jsonlines

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_path',
                        type=str,
                        help='path to jsonl data file')
    parser.add_argument('output_path',
                        type=str,
                        help='path to output file')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    data = []
    with jsonlines.open(args.data_path) as reader:
        for line in reader:
            if line['label']:
                if line['label'][0] in ["poetry", "poetry_line_continuation"]:
                    label = 1
                else:
                    label = 0
                data.append({
                    "window": line['window'],
                    "label": label
                })

    with jsonlines.open(args.output_path, 'w') as f:
        f.write_all(data)
