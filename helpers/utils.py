import re
import os
import zipfile
import json
from itertools import chain, islice


def load_metadata(archive_path):
    zip_file = zipfile.ZipFile(archive_path)
    with zip_file.open('gutenberg-dammit-files/gutenberg-metadata.json') as f:
        return json.load(f)


def gd_metadata_iter(data, field="Subject", pattern=r'^Poet.*', english_only=True):
    for item in data:
        matches_pattern = any([re.match(pattern, subj) for subj in item.get(field, [])])
        is_english = "English" in item.get("Language")
        if matches_pattern and (is_english or not english_only):
            yield item


def iter_examples(archive_path, metadata_etnry):
    prev_lines = ['', '', '']
    for line in iter_lines_from_gd_path(archive_path, metadata_entry.get('gd-path')):
        text = " [SEP] ".join(prev_lines) + " [SEP] " + line
        ex = {'text': text, 'line': line}
        prev_lines.pop(0)
        prev_lines.append(line)
        yield ex


def windowed_iter(items, n):
    items = (['BOS'] * n) + items + (['EOS'] * n)
    yield from zip(*[items[i:] for i in range((2*n)+1)])


def batch_iterable(iterable, n: int):
    """Return an iterable of batches with size n"""
    it = iter(iterable)
    for first in it:
        yield list(chain([first], islice(it, n-1)))


def iter_lines_from_gd_path(archive_path, gd_path):
    zip_file = zipfile.ZipFile(archive_path)
    with zip_file.open(os.path.join('gutenberg-dammit-files/', gd_path)) as f:
        for line in f.read().decode('utf-8').split('\n'):
            yield line.strip()
