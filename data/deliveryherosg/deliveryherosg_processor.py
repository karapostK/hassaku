import argparse
import os
import shutil

import pandas as pd

from data.data_utils import LOG_FILT_DATA_PATH, print_and_log, download_delivery_hero_sg_dataset

"""
N.B. The following pre-processing of the dataset is only performed to be in-line with the other experiments in this framework. 
The authors of the dataset mention specific challenges https://dl.acm.org/doi/10.1145/3604915.3610242 that need to be faced with the dataset
which can be incorporated in a future re-iteration of the framework. 

"""
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
    download_delivery_hero_sg_dataset('./')

if os.path.exists('./processed_dataset'):
    shutil.rmtree('./processed_dataset')
os.mkdir('./processed_dataset')

orders_train_path = './raw_dataset/orders_sg_train.txt'
orders_test_path = './raw_dataset/orders_sg_test.txt'

log_filt_data_file = open(os.path.join('./processed_dataset', LOG_FILT_DATA_PATH), 'w+')

train_lhs = pd.read_csv(orders_train_path, index_col=0).rename(columns={'customer_id': 'user', 'product_id': 'item'})
test_lhs = pd.read_csv(orders_test_path, index_col=0).rename(columns={'customer_id': 'user', 'product_id': 'item'})

print_and_log(log_filt_data_file, len(train_lhs), train_lhs.user.nunique(), train_lhs.item.nunique(),
              'Original Train Data')
print_and_log(log_filt_data_file, len(test_lhs), test_lhs.user.nunique(), test_lhs.item.nunique(), 'Original Test Data')

# Casting the day filed
train_lhs.order_day = train_lhs.order_day.apply(lambda x: int(x.split(' ')[0]))
test_lhs.order_day = test_lhs.order_day.apply(lambda x: int(x.split(' ')[0]))

# Generating the validation data. Following the same structure of the test data (taking 14 days before test data)
val_data_min_day = train_lhs.order_day.max() - 14

val_lhs = train_lhs[train_lhs.order_day >= val_data_min_day]
train_lhs = train_lhs[train_lhs.order_day < val_data_min_day]

print_and_log(log_filt_data_file, len(train_lhs), train_lhs.user.nunique(), train_lhs.item.nunique(),
              'Train Data without Validation Data')
print_and_log(log_filt_data_file, len(val_lhs), val_lhs.user.nunique(), val_lhs.item.nunique(),
              'Val Data')

# Casting the problem into product discovery recommendations.
# We keep only the first user-product interaction

train_lhs = train_lhs.sort_values(['order_day', 'order_time'])
val_lhs = val_lhs.sort_values(['order_day', 'order_time'])
test_lhs = test_lhs.sort_values(['order_day', 'order_time'])

train_lhs = train_lhs.drop_duplicates(subset=['user', 'item'])
val_lhs = val_lhs.drop_duplicates(subset=['user', 'item'])
test_lhs = test_lhs.drop_duplicates(subset=['user', 'item'])

print_and_log(log_filt_data_file, len(train_lhs), train_lhs.user.nunique(), train_lhs.item.nunique(),
              'Train Only First Interaction')
print_and_log(log_filt_data_file, len(val_lhs), val_lhs.user.nunique(), val_lhs.item.nunique(),
              'Val Only First Interaction')
print_and_log(log_filt_data_file, len(test_lhs), test_lhs.user.nunique(), test_lhs.item.nunique(),
              'Test Only First Interaction')

# Removing already discovered items from the val and test data

val_lhs = val_lhs.merge(train_lhs[['user', 'item']], on=['user', 'item'], how='left', indicator=True)
val_lhs = val_lhs[val_lhs['_merge'] == 'left_only'].drop(columns=['_merge'])

# For test data need to remove also data from validation
test_lhs = test_lhs.merge(train_lhs[['user', 'item']], on=['user', 'item'], how='left', indicator=True)
test_lhs = test_lhs[test_lhs['_merge'] == 'left_only'].drop(columns=['_merge'])

test_lhs = test_lhs.merge(val_lhs[['user', 'item']], on=['user', 'item'], how='left', indicator=True)
test_lhs = test_lhs[test_lhs['_merge'] == 'left_only'].drop(columns=['_merge'])

print_and_log(log_filt_data_file, len(val_lhs), val_lhs.user.nunique(), val_lhs.item.nunique(),
              'Only Interactions that Occur in Val')
print_and_log(log_filt_data_file, len(test_lhs), test_lhs.user.nunique(), test_lhs.item.nunique(),
              'Only Interactions that Occur in Test')

# Dropping users/items appearing only in val/test (pure cf recommendation)

val_users_in_train = set(val_lhs.user).intersection(train_lhs.user)
val_items_in_train = set(val_lhs.item).intersection(train_lhs.item)

test_users_in_train = set(test_lhs.user).intersection(train_lhs.user)
test_items_in_train = set(test_lhs.item).intersection(train_lhs.item)

val_lhs = val_lhs[(val_lhs.item.isin(val_items_in_train)) & (val_lhs.user.isin(val_users_in_train))]
test_lhs = test_lhs[(test_lhs.item.isin(test_items_in_train)) & (test_lhs.user.isin(test_users_in_train))]

print_and_log(log_filt_data_file, len(val_lhs), val_lhs.user.nunique(), val_lhs.item.nunique(),
              'Val Data, only users/items appearing at least once in Train')
print_and_log(log_filt_data_file, len(test_lhs), test_lhs.user.nunique(), test_lhs.item.nunique(),
              'Test Data, only users/items appearing at least once in Train')

# Assigning ids

# Defining a unique order for the index assignment
train_lhs = train_lhs.sort_values(['order_day', 'order_time', 'user'])

user_idxs = train_lhs.user.drop_duplicates().reset_index(drop=True)
item_idxs = train_lhs.item.drop_duplicates().reset_index(drop=True)

user_idxs.index.name = 'user_idx'
item_idxs.index.name = 'item_idx'

user_idxs = user_idxs.reset_index()
item_idxs = item_idxs.reset_index()

train_lhs = train_lhs.merge(user_idxs).merge(item_idxs)
val_lhs = val_lhs.merge(user_idxs).merge(item_idxs)
test_lhs = test_lhs.merge(user_idxs).merge(item_idxs)

print_and_log(log_filt_data_file, len(train_lhs), train_lhs.user.nunique(), train_lhs.item.nunique(),
              'Train Final')
print_and_log(log_filt_data_file, len(val_lhs), val_lhs.user.nunique(), val_lhs.item.nunique(),
              'Val Final')
print_and_log(log_filt_data_file, len(test_lhs), test_lhs.user.nunique(), test_lhs.item.nunique(),
              'Test Final')
log_filt_data_file.close()

# Saving locally
print('Saving data to ./processed_dataset')

train_lhs.to_csv('./processed_dataset/listening_history_train.csv', index=False)
val_lhs.to_csv('./processed_dataset/listening_history_val.csv', index=False)
test_lhs.to_csv('./processed_dataset/listening_history_test.csv', index=False)

user_idxs.to_csv('./processed_dataset/user_idxs.csv', index=False)
item_idxs.to_csv('./processed_dataset/item_idxs.csv', index=False)
