import zipfile

def filtered_iter(data, field, pattern, english_only=True):
    for item in data:
        matches_pattern = any([re.match(pattern, subj) for subj in item.get(field, [])])
        is_english = "English" in item.get("Language")
        if matches_pattern and (is_english or not english_only):
            yield item

def main():
    zip_file = zipfile.ZipFile(archive_path)
    with zip_file.open('gutenberg-dammit-files/gutenberg-metadata.json') as f:
        gutenberg_metadata = json.load(f)
    filtered_items = filtered_iter(gutenberg_metadata, "Subject", r'^Poet.*')
