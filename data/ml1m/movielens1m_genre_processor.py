import pandas as pd

movies = pd.read_csv('./raw_dataset/movies.dat', sep='::', engine='python',
                     names=['item', 'title', 'genres'], encoding='iso-8859-1')
items = pd.read_csv('./processed_dataset/item_idxs.csv')

# Extracting Genres
genres_set = set()
for movie_genre_list in movies.genres.str.split('|'):
    genres_set.update(movie_genre_list)

genres = sorted(list(genres_set))
n_genres = len(genres)

# Create genre id
genre_idxs = pd.DataFrame(enumerate(genres), columns=['genre_idx', 'genre'])
genre_to_id = genre_idxs.set_index('genre').genre_idx

# Create item_genre file
records = []

def extract_item_to_genre_mappings(row):
    item_idx = row['item_idx']
    item_genres = row['genres'].split('|')
    item_genre_ids = list(genre_to_id[item_genres])
    for item_genre_id in item_genre_ids:
        records.append((item_idx, item_genre_id))


movies.merge(items).apply(extract_item_to_genre_mappings, axis=1)

item_genre_idxs = pd.DataFrame(records, columns=['item_idx', 'genre_idx'])

genre_idxs.to_csv('./processed_dataset/genre_idxs.csv', index=False)
item_genre_idxs.to_csv('./processed_dataset/item_genre_idxs.csv', index=False)
