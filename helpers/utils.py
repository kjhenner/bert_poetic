import re
import os
from itertools import chain, islice


def filtered_iter(data, field, pattern, english_only=True):
    for item in data:
        matches_pattern = any([re.match(pattern, subj) for subj in item.get(field, [])])
        is_english = "English" in item.get("Language")
        if matches_pattern and (is_english or not english_only):
            yield item


def windowed_iter(items, n):
    items = (['BOS'] * n) + items + (['EOS'] * n)
    yield from zip(*[items[i:] for i in range((2*n)+1)])


def batch_iterable(iterable, n: int):
    """Return an iterable of batches with size n"""
    it = iter(iterable)
    for first in it:
        yield list(chain([first], islice(it, n-1)))


def zips_to_lines(zip_file, metadata_iter, include_metadata=False):
    for metadata_entry in metadata_iter:
        with zip_file.open(os.path.join('gutenberg-dammit-files/', metadata_entry.get('gd-path'))) as f:
            for line in f.read().decode('utf-8').split('\n'):
                if include_metadata:
                    yield line.strip(), metadata_entry
                else:
                    yield line.strip()
