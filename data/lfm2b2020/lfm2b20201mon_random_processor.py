import argparse
import os
import shutil

import pandas as pd

from data.data_utils import LOG_FILT_DATA_PATH, print_and_log, download_lfm2b_2020_dataset, k_core_filtering, \
    create_index, split_temporal_order_ratio_based, split_random_order_ratio_based

parser = argparse.ArgumentParser()

parser.add_argument('--force_download', '-d', action='store_true',
                    help='Whether or not to re-download the dataset if "raw_dataset" folder is detected. Default to '
                         'False',
                    default=False)

args = parser.parse_args()
force_download = args.force_download

if not os.path.exists('./raw_dataset') or force_download:
    if force_download and os.path.exists('./raw_dataset'):
        shutil.rmtree('./raw_dataset')
    download_lfm2b_2020_dataset('./')

if os.path.exists('./processed_dataset'):
    shutil.rmtree('./processed_dataset')
os.mkdir('./processed_dataset')

users_data_path = './raw_dataset/users.tsv'
listening_events_path = './raw_dataset/inter_dataset.tsv'
log_filt_data_file = open(os.path.join('./processed_dataset', LOG_FILT_DATA_PATH), 'w+')

lhs = pd.read_csv(listening_events_path, sep='\t', names=['user', 'item', 'timestamp'], skiprows=1, usecols=[0, 1, 3])
print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(), 'Original Data')

# Loading users
users = pd.read_csv(users_data_path, delimiter='\t', names=['user', 'gender'], usecols=[0, 3], skiprows=1)

# Only users with gender in m/f
users = users[(users.gender.isin(['m', 'f']))]
lhs = lhs[lhs.user.isin(set(users.user))]
print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(), 'Only users that reported m/f')

# Keeping the data only from the last month
lhs.timestamp = pd.to_datetime(lhs.timestamp)
lhs = lhs[lhs.timestamp >= '2020-02-20 00:00:00']
print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(), 'Only last month')

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

lhs = k_core_filtering(lhs, 5)

print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(),
              '5-core filtering')

lhs, user_idxs, item_idxs = create_index(lhs)

print('Splitting the data, random ordered - ratio-based (80-10-10)')

lhs, train_data, val_data, test_data = split_random_order_ratio_based(lhs,seed=1000)

print_and_log(log_filt_data_file, len(train_data), train_data.user.nunique(), train_data.item.nunique(), 'Train Data')
print_and_log(log_filt_data_file, len(val_data), val_data.user.nunique(), val_data.item.nunique(), 'Val Data')
print_and_log(log_filt_data_file, len(test_data), test_data.user.nunique(), test_data.item.nunique(), 'Test Data')

log_filt_data_file.close()

# Adding grouping information
user_idxs = user_idxs.merge(users)
user_idxs['group_idx'] = (user_idxs.gender == 'f').astype(int)  # 0 is Male 1 is Female

# Saving locally
print('Saving data to ./processed_dataset')

lhs.to_csv('./processed_dataset/listening_history.csv', index=False)
train_data.to_csv('./processed_dataset/listening_history_train.csv', index=False)
val_data.to_csv('./processed_dataset/listening_history_val.csv', index=False)
test_data.to_csv('./processed_dataset/listening_history_test.csv', index=False)

user_idxs.to_csv('./processed_dataset/user_idxs.csv', index=False)
item_idxs.to_csv('./processed_dataset/item_idxs.csv', index=False)
