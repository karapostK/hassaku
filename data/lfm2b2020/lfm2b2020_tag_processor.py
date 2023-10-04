import pandas as pd

tracks = pd.read_json('./raw_dataset/tags_micro_genre.json', lines=True)[['_id', 'tags']]
items = pd.read_csv('./processed_dataset/item_idxs.csv')

tagged_items = items.merge(tracks, left_on='item', right_on='_id')

# Extracting Genres
tags_set = set(tagged_items.tags.explode())

tags = sorted(list(tags_set))
n_tags = len(tags)

# Create tag id
tag_idxs = pd.DataFrame(enumerate(tags), columns=['tag_idx', 'tag'])
tag_to_id = tag_idxs.set_index('tag').tag_idx

# Create item_tag file
records = []


def extract_item_to_tag_mappings(row):
    item_idx = row['item_idx']
    item_tags = row['tags']
    item_tag_ids = list(tag_to_id[item_tags])
    for item_genre_id in item_tag_ids:
        records.append((item_idx, item_genre_id))


tagged_items.apply(extract_item_to_tag_mappings, axis=1)

item_tag_idxs = pd.DataFrame(records, columns=['item_idx', 'tag_idx'])

tag_idxs.to_csv('./processed_dataset/tag_idxs.csv', index=False)
item_tag_idxs.to_csv('./processed_dataset/item_tag_idxs.csv', index=False)
