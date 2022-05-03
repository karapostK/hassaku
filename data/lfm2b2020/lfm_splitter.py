import argparse
import math
import os

import pandas as pd
from tqdm import tqdm

from consts.consts import INF_STR, LOG_FILT_DATA_PATH
# INF_STR = "{:10d} entries {:7d} users {:7d} items for {}"
# LOG_FILT_DATA_PATH = "log_filtering_data.txt"

MIN_ITEM_INTER = 5
MIN_USER_INTER = 5


def print_and_log(log_file, n_lhs, n_users, n_items, text):
    info_string = INF_STR.format(n_lhs, n_users, n_items, text)
    log_file.write(info_string + '\n')
    print(info_string)


parser = argparse.ArgumentParser()

parser.add_argument("--listening_history_path", "-lh", type=str,
                    help="Path to 'listening_events.tsv' of the LFM dataset.")

parser.add_argument("--user_demo_file", "-ud", type=str,
                    help="Path to .tsv file containing user information.")

parser.add_argument("--saving_path", "-s", type=str, help="Path where to save the split data. Default to './'",
                    default='./')

args = parser.parse_args()

listening_history_path = args.listening_history_path
saving_path = args.saving_path
user_demo_file = args.user_demo_file

log_filt_data_file = open(LOG_FILT_DATA_PATH, 'w+')

df = pd.read_csv(user_demo_file, sep='\t')
n_user_bef = len(df)
df = df[df["Gender"].isin({"m", "f"})]
n_user_after = len(df)
good_user_ids = set(df["UserID"].unique())

lhs = pd.read_csv(listening_history_path, sep='\t')

print(f"Dropping {n_user_bef - n_user_after} users, as no gender information is available for them.")
lhs = lhs[lhs["user_id"].isin(good_user_ids)]

print_and_log(log_filt_data_file, len(lhs), lhs["user_id"].nunique(), lhs["item_id"].nunique(), 'Original Data')

lhs = lhs[(lhs.timestamp > "2020-01-01 00:00:00")]
print("Initial number of interactions:", len(lhs))

# n-core filtering
while True:
    start_number = len(lhs)

    # Item pass
    item_counts = lhs["item_id"].value_counts()
    item_above = set(item_counts[item_counts >= MIN_ITEM_INTER].index)
    lhs = lhs[lhs["item_id"].isin(item_above)]
    print('Records after item pass: ', len(lhs))

    # User pass
    user_counts = lhs["user_id"].value_counts()
    user_above = set(user_counts[user_counts >= MIN_USER_INTER].index)
    lhs = lhs[lhs["user_id"].isin(user_above)]
    print('Records after user pass: ', len(lhs))

    if len(lhs) == start_number:
        print('Exiting...')
        break

print_and_log(log_filt_data_file, len(lhs), lhs["user_id"].nunique(), lhs["item_id"].nunique(),
              f'{MIN_USER_INTER}u/{MIN_ITEM_INTER}i-core filtering')

lhs.reset_index(drop=True, inplace=True)
user_id_mapping = {uid: i for i, uid in enumerate(sorted(lhs["user_id"].unique()))}
item_id_mapping = {iid: i for i, iid in enumerate(sorted(lhs["item_id"].unique()))}
lhs["user_id"] = lhs["user_id"].map(user_id_mapping)
lhs["item_id"] = lhs["item_id"].map(item_id_mapping)

ui_old, ui_new = zip(*user_id_mapping.items())
df_user_mapping = pd.DataFrame({"user_id": ui_new, "user": ui_old})

ii_old, ii_new = zip(*item_id_mapping.items())
df_item_mapping = pd.DataFrame({"item_id": ii_new, "item": ii_old})
print('Splitting the data, temporal ordered - ratio-based (80-10-10)')

lhs = lhs.sort_values('timestamp')
train_idxs = []
val_idxs = []
test_idxs = []
for user, user_group in tqdm(lhs.groupby('user_id')):
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

print_and_log(log_filt_data_file, len(train_data), train_data["user_id"].nunique(), train_data["item_id"].nunique(),
              'Train Data')
print_and_log(log_filt_data_file, len(val_data), val_data["user_id"].nunique(), val_data["item_id"].nunique(),
              'Val Data')
print_and_log(log_filt_data_file, len(test_data), test_data["user_id"].nunique(), test_data["item_id"].nunique(),
              'Test Data')

log_filt_data_file.close()
# Saving locally

print('Saving data to {}'.format(saving_path))

train_data.to_csv(os.path.join(saving_path, 'listening_history_train.csv'), index=False)
val_data.to_csv(os.path.join(saving_path, 'listening_history_val.csv'), index=False)
test_data.to_csv(os.path.join(saving_path, 'listening_history_test.csv'), index=False)

df_user_mapping.to_csv(os.path.join(saving_path, "user_ids.csv"), index=False)
df_item_mapping.to_csv(os.path.join(saving_path, "item_ids.csv"), index=False)
