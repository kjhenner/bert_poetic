import json
import argparse
import zipfile
import re

from helpers.utils import filtered_iter, zips_to_lines

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a labeled poetry dataset based on Allison Parish's heuristic filter."
    )
    parser.add_argument(
        'archive_path',
        type=str,
        help='Path to the GutenbergDammit archive.'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='Path where the output will be saved'
    )
    return parser.parse_args()

def matches_aparrish_poetry_heuristics(prev_line, line):
    # Clean and filter logic from https://github.com/aparrish/gutenberg-poetry-corpus

    def clean(string):
        string = string.strip()
        match = re.search(r"( {3,}\d+\.?)$", string)
        if match:
            string = string[:match.start()]
        return re.sub(r"\[\d+\]", "", string)

    line = clean(line)
    prev_line = clean(prev_line)

    checks = {
        # between five and sixty-five characters (inclusive)
        'length': lambda prev, line: 5 <= len(line) <= 65,
        # not all upper-case
        'case': lambda prev, line: not(line.isupper()),
        # doesn't begin with a roman numeral
        'not_roman_numerals': lambda prev, line: \
                not(re.search("^[IVXDC]+\.", line)),
        # if the last line was long and this one is short, it's probably the end of
        # a paragraph
        'not_last_para_line': lambda prev, line: \
                not(len(prev) >= 65 and len(line) <= 65),
        # less than 25% of the line is punctuation characters
        'punct': lambda prev, line: \
            (len([ch for ch in line if ch.isalpha() or ch.isspace()]) / \
                (len(line)+0.01)) > 0.75,
        # doesn't begin with a bracket (angle or square)
        'no_bracket': lambda prev, line: \
                not(any([line.startswith(ch) for ch in '[<'])),
        # isn't in title case
        'not_title_case': lambda prev, line: not(line.istitle()),
        # isn't title case when considering only longer words
        'not_mostly_title_case': lambda prev, line: \
            not(" ".join([w for w in line.split() if len(w) >= 4]).istitle()),
        # not more than 50% upper-case characters
        'not_mostly_upper': lambda prev, line: \
            (len([ch for ch in line if ch.isupper()]) / (len(line)+0.01)) < 0.5,
        # doesn't begin or end with a digit
        'not_number': lambda prev, line: \
                not(re.search("^\d", line)) and not(re.search("\d$", line)),
        # passes the wordfilter
        #'wordfilter_ok': lambda prev, line: not(wordfilter.blacklisted(line))
        # Removed blacklisting for the classification stage of the pipeline.
        # It must be reinstated when training generative models so we can
        # be less likely to spew any hateful garbage!
    }
    return [name for name, check in checks.items() if not check(prev_line.strip(), line.strip())]


def main(args):
    zip_file = zipfile.ZipFile(args.archive_path)
    with zip_file.open('gutenberg-dammit-files/gutenberg-metadata.json') as f:
        gutenberg_metadata = json.load(f)
    filter_field = "Subject"
    pattern = r'^Poet.*'
    filtered_paths = filtered_iter(gutenberg_metadata, filter_field, pattern)
    items = zips_to_lines(zip_file, filtered_paths, include_metadata=True)
    with open(args.output_path, 'w') as f:
        prev_line = ''
        for line, metadata in items:
            matches = matches_aparrish_poetry_heuristics(prev_line, line)
            label = int(not any(matches))
            f.write(json.dumps({'line': line,
                                'prev_line': prev_line,
                                'label': label,
                                'gd-num': metadata['Num'],
                                'matches': matches}) + '\n')
            prev_line = line


if __name__ == "__main__":
    main(parse_args())
