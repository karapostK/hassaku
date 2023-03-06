import argparse
import os
import shutil

import pandas as pd

from data.data_utils import LOG_FILT_DATA_PATH, print_and_log, download_movielens_dataset, k_core_filtering, \
    create_index, split_temporal_order_ratio_based

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
    download_movielens_dataset('./', '10m')

if os.path.exists('./processed_dataset'):
    shutil.rmtree('./processed_dataset')
os.mkdir('./processed_dataset')

ratings_path = './raw_dataset/ratings.dat'
log_filt_data_file = open(os.path.join('./processed_dataset', LOG_FILT_DATA_PATH), 'w+')

lhs = pd.read_csv(ratings_path, sep='::', names=['user', 'item', 'rating', 'timestamp'], engine='python')

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

# Saving locally
print('Saving data to ./processed_dataset')

lhs.to_csv('./processed_dataset/listening_history.csv', index=False)
train_data.to_csv('./processed_dataset/listening_history_train.csv', index=False)
val_data.to_csv('./processed_dataset/listening_history_val.csv', index=False)
test_data.to_csv('./processed_dataset/listening_history_test.csv', index=False)

user_idxs.to_csv('./processed_dataset/user_idxs.csv', index=False)
item_idxs.to_csv('./processed_dataset/item_idxs.csv', index=False)
