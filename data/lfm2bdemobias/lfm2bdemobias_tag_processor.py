from collections import Counter

import pandas as pd
from tqdm import tqdm

from raw_dataset.tags import Tags

tracks = pd.read_csv('./raw_dataset/sampled_100000_items_tracks.txt', sep='\t', header=None, names=['artist', 'track'])
tracks['item'] = tracks.index
items = pd.read_csv('./processed_dataset/item_idxs.csv')
tracks = tracks.merge(items)

tag_mapper = Tags('./raw_dataset/tags.json.gz')

tag_counter = Counter()
for _, row in tqdm(tracks.iterrows()):
    artist_name = row['artist']
    track_name = row['track']
    track_tags = tag_mapper[artist_name, track_name]
    if track_tags is not None:
        tag_counter.update(track_tags.keys())

tags_set = {tag: count for tag, count in tag_counter.items() if count >= 50}.keys()
tags = sorted(list(tags_set))
n_tags = len(tags)

tag_idxs = pd.DataFrame(enumerate(tags), columns=['tag_idx', 'tag'])
tag_to_id = tag_idxs.set_index('tag').tag_idx

records = []


def extract_item_to_tag_mappings(row):
    item_idx = row['item_idx']
    artist_name = row['artist']
    track_name = row['track']
    track_tags = tag_mapper[artist_name, track_name]

    if track_tags is not None:
        track_tags = list(track_tags.keys() & tags_set)
        item_tag_ids = list(tag_to_id[track_tags])
        for item_genre_id in item_tag_ids:
            records.append((item_idx, item_genre_id))


tracks.apply(extract_item_to_tag_mappings, axis=1)

item_tag_idxs = pd.DataFrame(records, columns=['item_idx', 'tag_idx'])

tag_idxs.to_csv('./processed_dataset/tag_idxs.csv', index=False)
item_tag_idxs.to_csv('./processed_dataset/item_tag_idxs.csv', index=False)
