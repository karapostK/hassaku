import pandas as pd

movies = pd.read_csv('./raw_dataset/movies.dat', sep='::', engine='python',
                     names=['item', 'title', 'genres'], encoding='iso-8859-1')
items = pd.read_csv('./processed_dataset/item_idxs.csv')

# Extracting Genres
tags_set = set(movies.genres.str.split('|').explode())

tags = sorted(list(tags_set))
n_tags = len(tags)

# Create tag id
tag_idxs = pd.DataFrame(enumerate(tags), columns=['tag_idx', 'tag'])
tag_to_id = tag_idxs.set_index('tag').tag_idx

# Create item_tag file
records = []


def extract_item_to_tag_mappings(row):
    item_idx = row['item_idx']
    item_genres = row['genres'].split('|')
    item_tag_ids = list(tag_to_id[item_genres])
    for item_genre_id in item_tag_ids:
        records.append((item_idx, item_genre_id))


movies.merge(items).apply(extract_item_to_tag_mappings, axis=1)

item_tag_idxs = pd.DataFrame(records, columns=['item_idx', 'tag_idx'])

tag_idxs.to_csv('./processed_dataset/tag_idxs.csv', index=False)
item_tag_idxs.to_csv('./processed_dataset/item_tag_idxs.csv', index=False)
