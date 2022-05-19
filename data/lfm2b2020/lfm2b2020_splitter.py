### Full Dataset is available at: http://www.cp.jku.at/datasets/LFM-2b/

import argparse
import math
import os

import pandas as pd
from tqdm import tqdm

from data.data_utils import LOG_FILT_DATA_PATH, print_and_log

parser = argparse.ArgumentParser()

parser.add_argument('--listening_history_path', '-lh', type=str,
                    help="Path to 'users.tsv' and 'listening_events.tsv' of the LFM2b2020 dataset.")
parser.add_argument('--saving_path', '-s', type=str, help="Path where to save the split data. Default to './'",
                    default='./')

args = parser.parse_args()

listening_history_path = args.listening_history_path
saving_path = args.saving_path

user_info_path = os.path.join(listening_history_path, 'users.tsv')
listening_events_path = os.path.join(listening_history_path, 'listening_events.tsv')
log_filt_data_file = open(LOG_FILT_DATA_PATH, 'w+')

lhs = pd.read_csv(listening_events_path, sep='\t', names=['user', 'item', 'timestamp'], skiprows=1, usecols=[0, 1, 3])
print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(), 'Original Data')

# Loading users
users = pd.read_csv(user_info_path, delimiter='\t', names=['user', 'gender'], usecols=[0, 3], skiprows=1)

# Only users with gender in m/f
users = users[(users.gender.isin(['m', 'f']))]
lhs = lhs[lhs.user.isin(set(users.user))]
print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(), 'Only users that reported m/f')

# Keeping only interactions that have happened more than 1.
lhs_count = lhs.value_counts(subset=['user', 'item'])
lhs_count = lhs_count[lhs_count > 1]
lhs = lhs.set_index(['user', 'item']).loc[lhs_count.index]
lhs = lhs.reset_index()
print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(),
              'Only interactions that happened at least twice')

# Keeping only the first interaction
lhs = lhs.sort_values('timestamp')
lhs = lhs.drop_duplicates(subset=['user', 'item'])
print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(), 'Only first interaction')

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
user_idxs = lhs.user.drop_duplicates().reset_index(drop=True)
item_idxs = lhs.item.drop_duplicates().reset_index(drop=True)
user_idxs.index.name = 'user_idx'
item_idxs.index.name = 'item_idx'
user_idxs = user_idxs.reset_index()
item_idxs = item_idxs.reset_index()
lhs = lhs.merge(user_idxs).merge(item_idxs)

print('Splitting the data, temporal ordered - ratio-based (80-10-10)')

lhs = lhs.sort_values('timestamp')
train_idxs = []
val_idxs = []
test_idxs = []
for user, user_group in tqdm(lhs.groupby('user')):
    # Data is already sorted by timestamp
    n_val = n_test = math.ceil(len(user_group) * 0.1)
    n_train = len(user_group) - n_val - n_test

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

lhs.to_csv(os.path.join(saving_path, 'listening_history.csv'), index=False)
train_data.to_csv(os.path.join(saving_path, 'listening_history_train.csv'), index=False)
val_data.to_csv(os.path.join(saving_path, 'listening_history_val.csv'), index=False)
test_data.to_csv(os.path.join(saving_path, 'listening_history_test.csv'), index=False)

user_idxs.to_csv(os.path.join(saving_path, 'user_idxs.csv'), index=False)
item_idxs.to_csv(os.path.join(saving_path, 'item_idxs.csv'), index=False)
