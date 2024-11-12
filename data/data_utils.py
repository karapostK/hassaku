import bz2
import enum
import logging
import math
import os.path
import shutil
import zipfile

import gdown
import numpy as np
import pandas as pd
import requests
import scipy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataloader import TrainDataLoader, NegativeSampler
from data.dataset import TrainRecDataset, FullEvalDataset, ECFTrainRecDataset

LOG_FILT_DATA_PATH = "log_filtering_data.txt"

MOVIELENS_100K_DATASET_LINK = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_1M_DATASET_LINK = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_10M_DATASET_LINK = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"

LFM2B_2020_INTER_DATASET_LINK = "http://www.cp.jku.at/datasets/LFM-2b/recsys22/listening_events.tsv.bz2"
LFM2B_2020_USER_DATASET_LINK = "http://www.cp.jku.at/datasets/LFM-2b/recsys22/users.tsv.bz2"
LFM2B_2020_TRACK_DATASET_LINK = "http://www.cp.jku.at/datasets/LFM-2b/recsys22/tracks.tsv.bz2"

AMAZONVID2018_DATASET_LINK = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games.csv"

DELIVERY_HERO_SINGAPORE_DATASET_FILE_ID = "1v-FfCbLtv02EpNpopDx25EQnHZeT1nL2"


class DatasetsEnum(enum.Enum):
    """
    Enum to keep track of all the dataset available. Note that the name of the dataset  should correspond to a folder
    in data. e.g. ml1m has a corresponding folder in /data
    """
    ml1m = enum.auto()
    ml10m = enum.auto()
    amazonvid2018 = enum.auto()
    lfm2b2020 = enum.auto()
    deliveryherosg = enum.auto()
    lfm2bdemobias = enum.auto()
    deezer = enum.auto()


def print_and_log(log_file, n_lhs, n_users, n_items, text):
    """
    Prints to screen and logs to file statistics of the data during the processing steps.
    @param log_file: Name of the file to log information to
    @param n_lhs: Number of listening events
    @param n_users: Number of users
    @param n_items: Number of items
    @param text: Text to be added
    @return:
    """
    info_string = "{:10d} entries {:7d} users {:7d} items for {}".format(n_lhs, n_users, n_items, text)
    log_file.write(info_string + '\n')
    print(info_string)


def download_movielens_dataset(save_path: str = './', which: str = '1m'):
    """
    Downloads a movielens dataset.

    @type save_path: path to the folder where to save the raw dataset. Default to "./"
    @param which: Which movielens dataset should be donwloaded.
    """
    assert which in ['100k', '1m',
                     '10m'], f'The current implementation manages only 1m and 10m! {which} is not valid value.'

    # Downloading
    if which == '100k':
        url = MOVIELENS_100K_DATASET_LINK
    elif which == '1m':
        url = MOVIELENS_1M_DATASET_LINK
    elif which == '10m':
        url = MOVIELENS_10M_DATASET_LINK
    else:
        raise ValueError(f'The current implementation manages only 100k, 1m and 10m! {which} is not valid value.')

    print("Downloading the dataset...")
    req = requests.get(url)
    dataset_zip_name = os.path.join(save_path, "dataset.zip")

    with open(dataset_zip_name, 'wb') as fw:
        fw.write(req.content)

    # Unzipping
    with zipfile.ZipFile(dataset_zip_name, 'r') as zipr:
        zipr.extractall(save_path)

    os.remove(dataset_zip_name)

    if which == '100k':
        os.rename('ml-100k', 'raw_dataset')
    elif which == '1m':
        os.rename('ml-1m', 'raw_dataset')
    elif which == '10m':
        os.rename('ml-10M100K', 'raw_dataset')

    print('Dataset downloaded')


def download_lfm2b_2020_dataset(save_path: str = './'):
    """
    Downloads the LFM2b 2020 Subset
    @type save_path: path to the folder where to save the raw dataset. Default to "./"
    """

    if not os.path.exists(os.path.join(save_path, 'raw_dataset')):
        os.makedirs(os.path.join(save_path, 'raw_dataset'))
    # Downloading
    print("Downloading the dataset...")
    print('Downloading interaction data...')
    req = requests.get(LFM2B_2020_INTER_DATASET_LINK)
    data = bz2.decompress(req.content)
    file_name = os.path.join(save_path, "raw_dataset", "inter_dataset.tsv")
    with open(file_name, 'wb') as fw:
        fw.write(data)

    print('Downloading user data...')
    req = requests.get(LFM2B_2020_USER_DATASET_LINK)
    data = bz2.decompress(req.content)
    file_name = os.path.join(save_path, "raw_dataset", "users.tsv")
    with open(file_name, 'wb') as fw:
        fw.write(data)

    print('Downloading track data...')
    req = requests.get(LFM2B_2020_TRACK_DATASET_LINK)
    data = bz2.decompress(req.content)
    file_name = os.path.join(save_path, "raw_dataset", "tracks.tsv")

    with open(file_name, 'wb') as fw:
        fw.write(data)


def download_delivery_hero_sg_dataset(save_path: str = './'):
    """
    Downloads the Delivery Hero 2023 Dataset for Singapore
    https://dl.acm.org/doi/10.1145/3604915.3610242
    @type save_path: path to the folder where to save the raw dataset. Default to "./"
    """

    if not os.path.exists(os.path.join(save_path, 'raw_dataset')):
        os.makedirs(os.path.join(save_path, 'raw_dataset'))

    # Downloading
    print("Downloading the dataset...")
    dataset_zip_name = os.path.join(save_path, 'data_sg.zip')
    gdown.download(id=DELIVERY_HERO_SINGAPORE_DATASET_FILE_ID, output=dataset_zip_name)

    # Unzipping
    with zipfile.ZipFile(dataset_zip_name, 'r') as zipr:
        zipr.extractall(save_path)

    os.remove(dataset_zip_name)
    shutil.rmtree(os.path.join(save_path, '__MACOSX'))

    os.rename('data_sg', 'raw_dataset')

    print('Dataset downloaded')


def download_amazonvid2018_dataset(save_path: str = './'):
    """
    Downloads the Amazon 2018 VideoGame dataset
    @type save_path: path to the folder where to save the raw dataset. Default to "./"
    """

    if not os.path.exists(os.path.join(save_path, 'raw_dataset')):
        os.makedirs(os.path.join(save_path, 'raw_dataset'))

    # Downloading
    print("Downloading the dataset...")

    req = requests.get(AMAZONVID2018_DATASET_LINK, verify=False)
    file_name = os.path.join(save_path, "raw_dataset", "Video_Games.csv")
    with open(file_name, 'wb') as fw:
        fw.write(req.content)


def k_core_filtering(lhs: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Performs core-filtering on the dataset.
    @param lhs: Pandas Dataframe containing the listening records. Has columns ["user", "item"]
    @param k: Threshold for performing the k-core filtering.
    @return: Filtered Dataframe
    """
    while True:
        start_number = len(lhs)

        # Item pass
        item_counts = lhs.item.value_counts()
        item_above = set(item_counts[item_counts >= k].index)
        lhs = lhs[lhs.item.isin(item_above)]
        print('Records after item pass: ', len(lhs))

        # User pass
        user_counts = lhs.user.value_counts()
        user_above = set(user_counts[user_counts >= k].index)
        lhs = lhs[lhs.user.isin(user_above)]
        print('Records after user pass: ', len(lhs))

        if len(lhs) == start_number:
            print('Exiting...')
            break
    return lhs


def create_index(lhs: pd.DataFrame):
    """
    Associate an index for each user and item after having performed filtering steps. In order to avoid confusion, it
    sorts the data by timestamp, user, and item before creating the indexes.
    @param lhs: Pandas Dataframe containing the listening records. Has columns ["timestamp", "user", "item"]
    @return:
        lhs: Pandas Dataframe containing the listening records, now with the user_idx and item_idx columns
        user_idxs: Pandas Dataframe containing the user to user_idx mapping
        item_idxs: Pandas Dataframe containing the item to item_idx mapping
    """
    # Defining a unique order for the index assignment
    if 'timestamp' in lhs.columns:
        lhs = lhs.sort_values(['timestamp', 'user', 'item'])
    else:
        lhs = lhs.sort_values(['user', 'item'])

    # Creating simple integer indexes used for sparse matrices
    user_idxs = lhs.user.drop_duplicates().reset_index(drop=True)
    item_idxs = lhs.item.drop_duplicates().reset_index(drop=True)
    user_idxs.index.name = 'user_idx'
    item_idxs.index.name = 'item_idx'
    user_idxs = user_idxs.reset_index()
    item_idxs = item_idxs.reset_index()
    lhs = lhs.merge(user_idxs).merge(item_idxs)
    return lhs, user_idxs, item_idxs


def split_temporal_order_ratio_based(lhs: pd.DataFrame, ratios=(0.8, 0.1, 0.1)):
    """
    Split the interation time-wise, for each user, using the ratio specified as parameters. E.g. A split with (0.7,
    0.2,0.1) will first sort the data by timestamp then take the first 70% of the interactions as train data,
    the subsequent 20% as validation data, and the remaining last 10% as test data. In order to avoid confusion,
    it sorts the data by timestamp, user_idx, and item_idx before splitting.
    @param lhs: lhs: Pandas Dataframe
    containing the listening records. Has columns ["timestamp", "user_idx", "item_idx"]
    @param ratios: float values that denote the ratios for train, val, test. The values must sum to 1.
    @return:
        lhs: Pandas Dataframe containing the listening records sorted.
        train_data: Pandas Dataframe containing the train data.
        val_data: Pandas Dataframe containing the val data.
        test_data: Pandas Dataframe containing the test data.

    """
    assert sum(ratios) == 1, 'Ratios do not sum to 1!'

    lhs = lhs.sort_values(['timestamp', 'user', 'item'])
    train_idxs = []
    val_idxs = []
    test_idxs = []
    for user, user_group in tqdm(lhs.groupby('user')):
        # Data is already sorted by timestamp
        n_test = math.ceil(len(user_group) * ratios[-1])
        n_val = math.ceil(len(user_group) * ratios[-2])
        n_train = len(user_group) - n_val - n_test

        train_idxs += list(user_group.index[:n_train])
        val_idxs += list(user_group.index[n_train:n_train + n_val])
        test_idxs += list(user_group.index[-n_test:])

    train_data = lhs.loc[train_idxs]
    val_data = lhs.loc[val_idxs]
    test_data = lhs.loc[test_idxs]

    return lhs, train_data, val_data, test_data


def split_random_order_ratio_based(lhs: pd.DataFrame, ratios=(0.8, 0.1, 0.1), seed=13):
    """
    Split the interaction in a random fashion, for each user, using the ratio specified as parameters. E.g. A split with (0.7,
    0.2,0.1) will first randomize the data then take the first 70% of the interactions as train data,
    the subsequent 20% as validation data, and the remaining last 10% as test data.
    @param lhs: lhs: Pandas Dataframe
    containing the listening records. Has columns ["timestamp", "user_idx", "item_idx"]
    @param ratios: float values that denote the ratios for train, val, test. The values must sum to 1.
    @param seed: seed for the ranomization
    @return:
        lhs: Pandas Dataframe containing the listening records sorted.
        train_data: Pandas Dataframe containing the train data.
        val_data: Pandas Dataframe containing the val data.
        test_data: Pandas Dataframe containing the test data.

    """
    assert sum(ratios) == 1, 'Ratios do not sum to 1!'
    lhs = lhs.sample(frac=1., random_state=seed)
    train_idxs = []
    val_idxs = []
    test_idxs = []
    for user, user_group in tqdm(lhs.groupby('user')):
        n_test = math.ceil(len(user_group) * ratios[-1])
        n_val = math.ceil(len(user_group) * ratios[-2])
        n_train = len(user_group) - n_val - n_test

        train_idxs += list(user_group.index[:n_train])
        val_idxs += list(user_group.index[n_train:n_train + n_val])
        test_idxs += list(user_group.index[-n_test:])

    train_data = lhs.loc[train_idxs]
    val_data = lhs.loc[val_idxs]
    test_data = lhs.loc[test_idxs]

    return lhs, train_data, val_data, test_data


def get_dataloader(conf: dict, split_set: str) -> DataLoader:
    """
        Returns the dataloader associated to the configuration in conf
    """

    if split_set == 'train':
        if conf['alg'] == 'ecf':
            train_dataset_class = ECFTrainRecDataset
        else:
            train_dataset_class = TrainRecDataset

        train_dataset = train_dataset_class(data_path=conf['dataset_path'])

        sampler = NegativeSampler(
            train_dataset=train_dataset,
            n_neg=conf['neg_train'],
            neg_sampling_strategy=conf['train_neg_strategy']
        )
        dataloader = TrainDataLoader(
            sampler,
            train_dataset,
            batch_size=conf['train_batch_size'],
            shuffle=True,
            num_workers=conf['running_settings']['train_n_workers'],
            prefetch_factor=conf['running_settings']['prefetch_factor'] if 'prefetch_factor' in conf[
                'running_settings'] else 2
        )
        logging.info(f"Built Train DataLoader module \n"
                     f"- batch_size: {conf['train_batch_size']} \n"
                     f"- train_n_workers: {conf['running_settings']['train_n_workers']} \n")

    elif split_set == 'val':
        dataloader = DataLoader(
            FullEvalDataset(
                data_path=conf['dataset_path'],
                split_set='val',
            ),
            batch_size=conf['eval_batch_size'],
            num_workers=conf['running_settings']['eval_n_workers'],
        )
        logging.info(f"Built Val DataLoader module \n"
                     f"- batch_size: {conf['eval_batch_size']} \n"
                     f"- eval_n_workers: {conf['running_settings']['eval_n_workers']} \n")

    elif split_set == 'test':
        dataloader = DataLoader(
            FullEvalDataset(
                data_path=conf['dataset_path'],
                split_set='test',
            ),
            batch_size=conf['eval_batch_size'],
            num_workers=conf['running_settings']['eval_n_workers'],
        )
        logging.info(f"Built Test DataLoader module \n"
                     f"- batch_size: {conf['eval_batch_size']} \n"
                     f"- eval_n_workers: {conf['running_settings']['eval_n_workers']} \n")
    else:
        raise ValueError(f"split_set value '{split_set}' is invalid! Please choose from [train, val, test]")
    return dataloader


def build_user_and_item_tag_matrix(path_to_dataset_folder: str, alpha_smoothening: float = .01):
    """
    Builds the user x tag matrix and the item x tag matrix on the training data. For the user x tag matrix, each row
    represents the tag frequencies in that user train data. N.B. As multiple genres/tags can appear in an item, we
    perform row-wise normalization across the item-tag matrix **before** constructing the user-tag matrix. E.g. when a
    user watches a Western movie then their propensity (~in frequency) towards Western movies is increased by 1. When a
    user watches a Western|Sci-Fi movie then their propensity is split by both genres, effectively increasing 0.5 for
    Western and 0.5 for Sci-Fi. This procedure is equivalent to Harald Steck "Calibrated Recommendations" RecSys 2018.
    @param path_to_dataset_folder: Path to the dataset folder. Code will automatically fill out the rest,
    @param alpha_smoothening: alpha value used to smoothen the training distribution (eq. 7 in H. Steck Calibrated Recommendations)
    @return:
        - user_tag_matrix
        - item_tag_matrix
    """

    assert 0 <= alpha_smoothening <= 1, 'Alpha value out of bounds'

    # Load Tag Matrix & Training Data
    item_csv = pd.read_csv(os.path.join(path_to_dataset_folder, 'processed_dataset/item_idxs.csv'))
    user_csv = pd.read_csv(os.path.join(path_to_dataset_folder, 'processed_dataset/user_idxs.csv'))
    tag_csv = pd.read_csv(os.path.join(path_to_dataset_folder, 'processed_dataset/tag_idxs.csv'))
    item_tag_idxs_csv = pd.read_csv(os.path.join(path_to_dataset_folder, 'processed_dataset/item_tag_idxs.csv'))
    train_data = pd.read_csv(os.path.join(path_to_dataset_folder, 'processed_dataset/listening_history_train.csv'))[
        ['user_idx', 'item_idx']]

    n_tags = len(tag_csv)
    n_items = len(item_csv)
    n_users = len(user_csv)

    # Building Tag Matrix
    tag_matrix = torch.zeros(size=(n_items, n_tags), dtype=torch.float)
    tag_matrix[[item_tag_idxs_csv.item_idx, item_tag_idxs_csv.tag_idx]] = 1.

    # Normalizing row-wise
    tag_matrix /= tag_matrix.sum(-1)[:, None]
    tag_matrix[torch.isnan(tag_matrix)] = 0.  # Dealing with items with no tags

    # Building Train Matrix
    train_mtx = scipy.sparse.csr_matrix(
        (torch.ones(len(train_data), dtype=torch.int16), (train_data.user_idx, train_data.item_idx)),
        shape=(n_users, n_items)
    )

    # Computing User-Tag Frequencies
    users_tag_frequencies = train_mtx @ tag_matrix
    n_items_per_user = train_mtx.sum(-1).A
    users_tag_frequencies /= n_items_per_user

    # Smoothening (eq.7)
    users_tag_frequencies = alpha_smoothening / n_tags + (1 - alpha_smoothening) * users_tag_frequencies

    return torch.tensor(users_tag_frequencies), tag_matrix


def build_user_and_item_pop_matrix(path_to_dataset_folder: str, alpha_smoothening: float = .01):
    """
    Builds the user x pop matrix and the item x pop matrix on the training data.
    Bucketing follows: https://ceur-ws.org/Vol-3268/paper5.pdf and https://arxiv.org/pdf/2103.06364.pdf
    @param path_to_dataset_folder: Path to the dataset folder. Code will automatically fill out the rest,
    @param alpha_smoothening: alpha value used to smoothen the training distribution (eq. 7 in H. Steck Calibrated Recommendations)
    @return:
        - user_pop_matrix
        - item_pop_matrix
    """

    assert 0 <= alpha_smoothening <= 1, 'Alpha value out of bounds'

    # Load Training Data
    item_csv = pd.read_csv(os.path.join(path_to_dataset_folder, 'processed_dataset/item_idxs.csv'))
    user_csv = pd.read_csv(os.path.join(path_to_dataset_folder, 'processed_dataset/user_idxs.csv'))
    train_data = pd.read_csv(os.path.join(path_to_dataset_folder, 'processed_dataset/listening_history_train.csv'))[
        ['user_idx', 'item_idx']]

    n_items = len(item_csv)
    n_users = len(user_csv)
    train_mtx = scipy.sparse.csr_matrix(
        (torch.ones(len(train_data), dtype=torch.float), (train_data.user_idx, train_data.item_idx)),
        shape=(n_users, n_items))

    # --- Compute Item Popularity Matrix --- #
    items_pop = train_mtx.sum(0).A1  # [n_items]
    pop_mass = items_pop.sum()  # total number of interactions
    items_pop /= pop_mass  # Normalizing respect to the mass

    sorted_items_idxs = np.argsort(-items_pop)

    mtx_row_idx = []
    mtx_col_idx = []

    curr_pop_mass = 0

    end_top_threshold = 0.2
    end_middle_threshold = 0.8

    for item_idx in sorted_items_idxs:

        curr_pop_mass += items_pop[item_idx]
        mtx_row_idx.append(item_idx)

        if curr_pop_mass < end_top_threshold:
            mtx_col_idx.append(0)
        elif curr_pop_mass < end_middle_threshold:
            mtx_col_idx.append(1)
        else:
            mtx_col_idx.append(2)

    # Creating the Matrix
    items_pop_mtx = scipy.sparse.csr_matrix(
        (torch.ones(len(mtx_row_idx), dtype=torch.float), (mtx_row_idx, mtx_col_idx)),
        shape=(n_items, 3))

    # --- Computing User Popularity --- #

    user_pop_mtx = train_mtx @ items_pop_mtx
    user_pop_mtx = user_pop_mtx.A
    user_pop_mtx /= user_pop_mtx.sum(-1)[:, None]

    # Smoothening (eq.7)
    user_pop_mtx = alpha_smoothening / 3 + (1 - alpha_smoothening) * user_pop_mtx

    return torch.tensor(user_pop_mtx), torch.tensor(items_pop_mtx.A)
