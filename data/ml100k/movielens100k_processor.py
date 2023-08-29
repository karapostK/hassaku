import argparse
import os
import shutil

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

from data.data_utils import LOG_FILT_DATA_PATH, print_and_log, download_movielens_dataset, k_core_filtering, \
    create_index, split_temporal_order_ratio_based

parser = argparse.ArgumentParser()

parser.add_argument('--force_download', '-d', action='store_true',
                    help='Whether or not to re-download the dataset if "raw_dataset" folder is detected. Default to '
                         'False',
                    default=False)

parser.add_argument('--patience_gamma', '-g', action='store_true',
                    help='The value of the patience to be used when computing the exposure. Default to 0.5',
                    default=0.5)

args = parser.parse_args()
force_download = args.force_download
patience_gamma = args.patience_gamma

if not os.path.exists('./raw_dataset') or force_download:
    if force_download and os.path.exists('./raw_dataset'):
        shutil.rmtree('./raw_dataset')
    download_movielens_dataset('./', '100k')

if os.path.exists('./processed_dataset'):
    shutil.rmtree('./processed_dataset')
os.mkdir('./processed_dataset')

ratings_path = './raw_dataset/u.data'
user_info_path = './raw_dataset/u.user'
log_filt_data_file = open(os.path.join('./processed_dataset', LOG_FILT_DATA_PATH), 'w+')

lhs = pd.read_csv(ratings_path, sep='\t', names=['user', 'item', 'rating', 'timestamp'], engine='python')

print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(), 'Original Data')

# We keep only ratings above 3.5
lhs = lhs[lhs.rating >= 3.5]

print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(),
              'Only Positive Interactions (>= 3.5)')

lhs = k_core_filtering(lhs, 5)

print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(),
              '5-core filtering')

lhs, user_idxs, item_idxs = create_index(lhs)

print('Splitting the data, temporal ordered - ratio-based (80-10-10)')

lhs, train_data, val_data, test_data = split_temporal_order_ratio_based(lhs)

print_and_log(log_filt_data_file, len(train_data), train_data.user.nunique(), train_data.item.nunique(), 'Train Data')
print_and_log(log_filt_data_file, len(val_data), val_data.user.nunique(), val_data.item.nunique(), 'Val Data')
print_and_log(log_filt_data_file, len(test_data), test_data.user.nunique(), test_data.item.nunique(), 'Test Data')

log_filt_data_file.close()

# Adding grouping information
users = pd.read_csv(user_info_path, sep='|', usecols=[0, 2], names=['user', 'gender'], engine='python')
user_idxs = user_idxs.merge(users)
user_idxs['group_idx'] = (user_idxs.gender == 'F').astype(int)  # 0 is Male 1 is Female



# Saving locally
print('Saving data to ./processed_dataset')

lhs.to_csv('./processed_dataset/listening_history.csv', index=False)
train_data.to_csv('./processed_dataset/listening_history_train.csv', index=False)
val_data.to_csv('./processed_dataset/listening_history_val.csv', index=False)
test_data.to_csv('./processed_dataset/listening_history_test.csv', index=False)

# user_idxs.to_csv('./processed_dataset/user_idxs.csv', index=False)
# item_idxs.to_csv('./processed_dataset/item_idxs.csv', index=False)

# Adding item genre
movies = pd.read_csv('./raw_dataset/u.item', sep='|', encoding='latin-1',
                     names=['item', 'title', 'release', '||', 'link'] + list(range(19)))


movie_genres = movies.drop(columns=['title', 'release', '||', 'link'])
item_idxs = item_idxs.merge(movies)
item_idxs[['item_idx', 'item'] + list(range(19))].to_csv('./processed_dataset/item_idxs.csv', index=False)

# Get genre profile of users
genres_columns = [i for i in range(19)]

movies_genres_array = item_idxs[genres_columns].to_numpy()
movies_genres_array = movies_genres_array / np.linalg.norm(movies_genres_array, ord=1, axis=1).reshape(-1, 1)
movies_genres_coo = coo_matrix(movies_genres_array)

movies_genres_df = item_idxs.copy()
movies_genres_df[genres_columns] = movies_genres_array

train_rows = train_data['user_idx']
train_cols = train_data['item_idx']

train_coo = coo_matrix(([1] * len(train_rows), (train_rows, train_cols)))

users_genre_profile = train_coo.dot(movies_genres_coo).toarray()
users_genre_profile = users_genre_profile / np.linalg.norm(users_genre_profile, ord=1, axis=1).reshape(-1, 1)

user_idxs[genres_columns] = users_genre_profile
user_idxs.to_csv('./processed_dataset/user_idxs.csv', index=False)

# Get genre exposure of users
print('Computing user genre exposure')
user_genre_exposures = train_data.merge(item_idxs).drop(columns=['item', 'rating', 'timestamp', 'item_idx']).groupby(
    ['user', 'user_idx']).sum().reset_index()
user_genre_exposures['total'] = user_genre_exposures[list(range(19))].sum(axis=1)

user_genre_exposures['sum_positive_exposures'] = (patience_gamma - patience_gamma ** (
        user_genre_exposures['total'] + 1)) / (1 - patience_gamma)
user_genre_exposures['sum_negative_exposures'] = (patience_gamma ** (
        user_genre_exposures['total'] + 1) - patience_gamma ** (item_idxs.item.nunique() + 1)) / (1 - patience_gamma)

genres_count = item_idxs[list(range(19))].copy()
genres_count = genres_count.sum()
minus_columns_names = [str(i) + '_minus' for i in list(range(19))]

user_genre_exposures[minus_columns_names] = user_genre_exposures[range(19)].sub(genres_count, axis=1)

user_genre_exposures[list(range(19))] = user_genre_exposures[range(19)].mul(
    user_genre_exposures['sum_positive_exposures'] / user_genre_exposures['total'], axis=0)

user_genre_exposures[minus_columns_names] = user_genre_exposures[minus_columns_names].mul(
    user_genre_exposures['sum_negative_exposures'] / (item_idxs.item.nunique() - user_genre_exposures['total']), axis=0)

column_renaming = {i: minus_columns_names[i] for i in range(19)}

user_genre_exposures[list(range(19))] = user_genre_exposures[list(range(19))].rename(column_renaming, axis=1).sub(
    user_genre_exposures[minus_columns_names])

user_genre_exposures = user_genre_exposures.drop(
    columns=minus_columns_names + ['||', 'total', 'sum_positive_exposures', 'sum_negative_exposures'])
user_genre_exposures[list(range(19))] = user_genre_exposures[list(range(19))].div(genres_count)

user_genre_exposures[['user_idx', 'user'] + list(range(19))].to_csv('./processed_dataset/user_genre_exposure.csv', index=False)
