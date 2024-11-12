import os
import shutil

import pandas as pd

from data.data_utils import LOG_FILT_DATA_PATH, print_and_log, k_core_filtering, \
    create_index, split_random_order_ratio_based

if os.path.exists('./processed_dataset'):
    shutil.rmtree('./processed_dataset')
os.mkdir('./processed_dataset')

users_data_path = './raw_dataset/user_country.csv'
listening_events_path = './raw_dataset/DEEZER_GLOBAL.inter'
track_active_data_path = './raw_dataset/metadata_DEEZER_active.csv'
track_origin_data_path = './raw_dataset/metadata_DEEZER_origin.csv'
track_musicbrainz_data_path = './raw_dataset/metadata_DEEZER_musicbrainz.csv'

log_filt_data_file = open(os.path.join('./processed_dataset', LOG_FILT_DATA_PATH), 'w+')

lhs = pd.read_csv(listening_events_path, sep=',', names=['user', 'item'], usecols=[0, 1], skiprows=1)
print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(), 'Original Data')

# Dropping multiple interactions (casting into general recommendation)

lhs = lhs.drop_duplicates()
print_and_log(log_filt_data_file, len(lhs), lhs.user.nunique(), lhs.item.nunique(), 'Without Duplicates')

# Loading users
users = pd.read_csv(users_data_path, delimiter=',', names=['user', 'user_country'], skiprows=1)

# Loading track info
track_active = pd.read_csv(track_active_data_path, delimiter=',', names=['item', 'item_country'], skiprows=1)
track_origin = pd.read_csv(track_origin_data_path, delimiter=',', names=['item', 'item_country'], skiprows=1)
track_musicbrainz = pd.read_csv(track_musicbrainz_data_path, delimiter=',', names=['item', 'item_country'], skiprows=1)

# Dropping non-valid countries
track_active = track_active.loc[track_active['item_country'] != '0']
track_origin = track_origin.loc[track_origin['item_country'] != '0']
track_musicbrainz = track_musicbrainz.loc[track_musicbrainz['item_country'] != '0']

# Doing the several merges
lhs = lhs.merge(users, how='inner', on='user')

lhs_active = lhs.merge(track_active, how='inner', on='item')
lhs_origin = lhs.merge(track_origin, how='inner', on='item')
lhs_musicbrainz = lhs.merge(track_musicbrainz, how='inner', on='item')

print_and_log(log_filt_data_file, len(lhs_active), lhs_active.user.nunique(), lhs_active.item.nunique(),
              'Active Country')
print_and_log(log_filt_data_file, len(lhs_origin), lhs_origin.user.nunique(), lhs_origin.item.nunique(),
              'Origin Country')
print_and_log(log_filt_data_file, len(lhs_musicbrainz), lhs_musicbrainz.user.nunique(), lhs_musicbrainz.item.nunique(),
              'MusicBrainz Country')

# K-core filtering

lhs_active = k_core_filtering(lhs_active, 5)
lhs_origin = k_core_filtering(lhs_origin, 5)
lhs_musicbrainz = k_core_filtering(lhs_musicbrainz, 5)

print_and_log(log_filt_data_file, len(lhs_active), lhs_active.user.nunique(), lhs_active.item.nunique(),
              'Active Country - 5-core')
print_and_log(log_filt_data_file, len(lhs_origin), lhs_origin.user.nunique(), lhs_origin.item.nunique(),
              'Origin Country - 5-core')
print_and_log(log_filt_data_file, len(lhs_musicbrainz), lhs_musicbrainz.user.nunique(), lhs_musicbrainz.item.nunique(),
              'MusicBrainz Country - 5-core')

# Splitting the date per-user random items
for lhs_selected, lhs_type in zip([lhs_active, lhs_origin, lhs_musicbrainz], ['active', 'origin', 'musicbrainz']):
    for seed in [683, 21945, 1700]:
        lhs, user_idxs, item_idxs = create_index(lhs_selected)
        lhs, train_data, val_data, test_data = split_random_order_ratio_based(lhs, seed=seed)

        print_and_log(log_filt_data_file, len(train_data), train_data.user.nunique(), train_data.item.nunique(),
                      '{} - Train Data'.format(lhs_type))
        print_and_log(log_filt_data_file, len(val_data), val_data.user.nunique(), val_data.item.nunique(),
                      '{} - Val Data'.format(lhs_type))
        print_and_log(log_filt_data_file, len(test_data), test_data.user.nunique(), test_data.item.nunique(),
                      '{} - Test Data'.format(lhs_type))

        # Saving locally

        saving_path = './processed_dataset/{}_{}'.format(lhs_type, seed)
        print(f'Saving data to {saving_path}')

        if os.path.exists(saving_path):
            shutil.rmtree(saving_path)
        os.mkdir(saving_path)

        lhs.to_csv(os.path.join(saving_path, 'listening_history.csv'), index=False)
        train_data.to_csv(os.path.join(saving_path, 'listening_history_train.csv'), index=False)
        val_data.to_csv(os.path.join(saving_path, 'listening_history_val.csv'), index=False)
        test_data.to_csv(os.path.join(saving_path, 'listening_history_test.csv'), index=False)

        user_idxs.to_csv(os.path.join(saving_path, 'user_idxs.csv'), index=False)
        item_idxs.to_csv(os.path.join(saving_path, 'item_idxs.csv'), index=False)

log_filt_data_file.close()
