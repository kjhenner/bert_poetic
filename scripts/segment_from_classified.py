import argparse
import json
import jsonlines


def parse_args():
    parser = argparse.ArgumentParser(description='Segment a sequence of classified lines into distinct poems')

    parser.add_argument('data_path', type=str)
    #parser.add_argument('output_path', type=str)

    return parser.parse_args()


def segment(line_iter, non_poetry_line_threshold=4, header_line_max=15):
    in_poem = False
    segments = []
    header_buffer = []
    non_poetry_line_buffer = []
    poetry_line_buffer = []
    gd_num = -1
    for line in line_iter:
        if in_poem:
            if line['pred'] > 0.5 and gd_num == line['gd-num']:
                gd_num = line['gd-num']
                poetry_line_buffer += non_poetry_line_buffer
                non_poetry_line_buffer = []
                poetry_line_buffer.append(line)
            else:
                gd_num = line['gd-num']
                non_poetry_line_buffer.append(line)
                if len(non_poetry_line_buffer) > non_poetry_line_threshold:
                    if poetry_line_buffer:
                        segments.append({
                            'gd-num': poetry_line_buffer[0]['gd-num'],
                            'poetry_lines ': [item['line'] for item in poetry_line_buffer],
                            'header_lines ': [item['line'] for item in header_buffer[-header_line_max:]]
                        })
                        print(json.dumps(segments[-1], indent=2))
                    header_buffer = []
                    poetry_line_buffer = []
                    in_poem = False
        else:
            if line['pred'] > 0.5 and gd_num == line['gd-num']:
                gd_num = line['gd-num']
                in_poem = True
                header_buffer = non_poetry_line_buffer.copy()
                non_poetry_line_buffer = []
            else:
                gd_num = line['gd-num']
                non_poetry_line_buffer.append(line)


if __name__ == "__main__":
    args = parse_args()

    with jsonlines.open(args.data_path) as reader:
        segment(reader)
