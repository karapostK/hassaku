import argparse
import math
import os

import pandas as pd
from tqdm import tqdm

from consts.consts import INF_STR, LOG_FILT_DATA_PATH


def print_and_log(log_file, n_lhs, n_users, n_items, text):
    info_string = INF_STR.format(n_lhs, n_users, n_items, text)
    log_file.write(info_string + '\n')
    print(info_string)


parser = argparse.ArgumentParser()

parser.add_argument('--listening_history_path', '-lh', type=str,
                    help="Path to 'ratings.dat' of the Movielens1M dataset.")
parser.add_argument('--saving_path', '-s', type=str, help="Path where to save the split data. Default to './'",
                    default='./')

args = parser.parse_args()

listening_history_path = args.listening_history_path
saving_path = args.saving_path

ratings_path = os.path.join(listening_history_path, 'ratings.dat')
log_filt_data_file = open(LOG_FILT_DATA_PATH, 'w+')

lhs = pd.read_csv(ratings_path, sep='::', names=['user', 'item', 'rating', 'timestamp'])

print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(), 'Original Data')

# We keep only ratings above 3.5
lhs = lhs[lhs.rating >= 3.5]

print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(),
              'Only Positive Interactions (>= 3.5)')

# 5-core filtering
while True:
    start_number = len(lhs)

    # Item pass
    item_counts = lhs.item.value_counts()
    item_above = set(item_counts[item_counts >= 5].index)
    lhs = lhs[lhs.item.isin(item_above)]
    print('Records after item pass: ', len(lhs))

    # User pass
    user_counts = lhs.user.value_counts()
    user_above = set(user_counts[user_counts >= 5].index)
    lhs = lhs[lhs.user.isin(user_above)]
    print('Records after user pass: ', len(lhs))

    if len(lhs) == start_number:
        print('Exiting...')
        break

print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(),
              '5-core filtering')

# Creating simple integer indexes used for sparse matrices
user_ids = lhs.user.drop_duplicates().reset_index(drop=True)
item_ids = lhs.item.drop_duplicates().reset_index(drop=True)
user_ids.index.name = 'user_id'
item_ids.index.name = 'item_id'
user_ids = user_ids.reset_index()
item_ids = item_ids.reset_index()
lhs = lhs.merge(user_ids).merge(item_ids)

print('Splitting the data, temporal ordered - ratio-based (80-10-10)')

lhs = lhs.sort_values('timestamp')
train_idxs = []
val_idxs = []
test_idxs = []
for user, user_group in tqdm(lhs.groupby('user')):
    # Data is already sorted by timestamp
    if len(user_group) <= 10:
        # Not enough data for val/test data. Place the user in train.
        train_idxs += (list(user_group.index))
    else:
        n_train = math.ceil(len(user_group) * 0.8)
        n_val = math.ceil((len(user_group) - n_train) / 2)
        n_test = len(user_group) - n_train - n_val

        train_idxs += list(user_group.index[:n_train])
        val_idxs += list(user_group.index[n_train:n_train + n_val])
        test_idxs += list(user_group.index[-n_test:])

train_data = lhs.loc[train_idxs]
val_data = lhs.loc[val_idxs]
test_data = lhs.loc[test_idxs]

print_and_log(log_filt_data_file, len(train_data), train_data.user.nunique(), train_data.item.nunique(), 'Train Data')
print_and_log(log_filt_data_file, len(val_data), val_data.user.nunique(), val_data.item.nunique(), 'Val Data')
print_and_log(log_filt_data_file, len(test_data), test_data.user.nunique(), test_data.item.nunique(), 'Test Data')

log_filt_data_file.close()
# Saving locally

print('Saving data to {}'.format(saving_path))

train_data.to_csv(os.path.join(saving_path, 'listening_history_train.csv'), index=False)
val_data.to_csv(os.path.join(saving_path, 'listening_history_val.csv'), index=False)
test_data.to_csv(os.path.join(saving_path, 'listening_history_test.csv'), index=False)

user_ids.to_csv(os.path.join(saving_path, 'user_ids.csv'), index=False)
item_ids.to_csv(os.path.join(saving_path, 'item_ids.csv'), index=False)
