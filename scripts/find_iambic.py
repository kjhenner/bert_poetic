import zipfile
import json
import re
import os
import jsonlines


def filtered_iter(data, field, pattern, english_only=True):
    for item in data:
        matches_pattern = any([re.match(pattern, subj) for subj in item.get(field, [])])
        is_english = "English" in item.get("Language")
        if matches_pattern and (is_english or not english_only):
            yield item


def main():
    archive_path = "/mnt/atlas/gutenberg_dammit/gutenberg-dammit-files-v002.zip"
    zip_file = zipfile.ZipFile(archive_path)
    with zip_file.open('gutenberg-dammit-files/gutenberg-metadata.json') as f:
        gutenberg_metadata = json.load(f)

    filtered_items = filtered_iter(gutenberg_metadata, "Subject", r'^Poet.*')

    data = []
    for item in filtered_items:
        with zip_file.open(os.path.join('gutenberg-dammit-files', item.get('gd-path'))) as f:
            lines = f.read().decode('utf-8').split('\n')
        data.append({"title": item.get('Title'),
            "text": '\n'.join(lines[:50] + ["......."] + lines[500:550]),
                     "gd-path": item.get('gd-path')})
    with jsonlines.open('samples.jsonl', 'w') as writer:
        writer.write_all(data)



if __name__ == "__main__":
    main()
