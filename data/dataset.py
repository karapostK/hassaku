import os

import numpy as np
import pandas as pd
from scipy import sparse as sp
from torch.utils import data
from torch.utils.data.dataset import T_co


class RecDataset(data.Dataset):
    """
    This class is used to encapsulate a generic dataset of user x item interactions. The class is tailored for a
    user-based leave-one-out split where:
        - user-based means that for each user we withold part of the interactions as training, part as validation, and part as testing
        - leave-one-out means that one interaction is used as testing, another one for validation, and the rest of the data for training
    Even though it is not required by the code, the class has been used for time-sorted interactions in the leave-one-out strategy (
    a.k.a. the last iteraction is used for testing, the one before for validation, and the rest for training).

   To use this class for any dataset, please check how data should be split according to the pre-existing splitter functions (e.g. movielens_splitter.py)

    This class implements some basic functionalities about negative sampling. The negative sampling for a specific user
    is influenced by the split_set:
        - split_set = train: The other training items are excluded from the sampling.
        - split_set = val: The other validation items and training items are excluded from the sampling.
        - split_set = test: The other test items and training items are excluded from the sampling.

    About the data management and access:
    To perform a fast iteration and sampling over the dataset, we use two sparse matrices (COO and CSR). The COO
    is used for iteration over the training data while the CSR for fast negative sampling. We always load the train
    CSR since it is used to exclude the training data from the negative sampling also for Validation and Testing.
    NB. Depending on the split_set, the matrices may have different data. Train COO and Train CSR have always the
    same data. However, Val CSR has Val + Train data (same applies for test). This is due to the negative sampling
    in the csr matrix, for which we also exclude items from training (see below).
    """

    def __init__(self, data_path: str, split_set: str, n_neg: int, neg_strategy: str = 'uniform'):
        """
        :param data_path: path to the directory with the listening_history_*, item_ids, and user_ids files.
        :param split_set: Value in [train, val, test].
        :param n_neg: Number of negative samples.
        :param neg_strategy: Strategy to select the negative samples.
        """
        assert split_set in ['train', 'val', 'test'], f'<{split_set}> is not a valid value for split set!'

        self.data_path = data_path
        self.split_set = split_set
        self.n_neg = n_neg
        self.neg_strategy = neg_strategy

        self.n_users = None
        self.n_items = None

        self.item_ids = None

        self.iteration_matrix = None
        self.sampling_matrix = None

        self.pop_distribution = None

        self.load_data()

        print(f'Built RecDataset module \n'
              f'- data_path: {self.data_path} \n'
              f'- n_users: {self.n_users} \n'
              f'- n_items: {self.n_items} \n'
              f'- n_interactions: {self.iteration_matrix.nnz} \n'
              f'- split_set: {self.split_set} \n'
              f'- n_neg: {self.n_neg} \n'
              f'- neg_strategy: {self.neg_strategy} \n')

    def load_data(self):
        print('Loading data')

        user_ids = pd.read_csv(os.path.join(self.data_path, 'user_ids.csv'))
        item_ids = pd.read_csv(os.path.join(self.data_path, 'item_ids.csv'))

        self.n_users = len(user_ids)
        self.n_items = len(item_ids)

        train_lhs = pd.read_csv(os.path.join(self.data_path, 'listening_history_train.csv'))

        train_csr = sp.csr_matrix(
            (np.ones(len(train_lhs), dtype=np.int16), (train_lhs.user_id, train_lhs.item_id)),
            shape=(self.n_users, self.n_items))

        # Computing the popularity distribution (see _neg_sample_popular)
        item_popularity = np.array(train_csr.sum(axis=0)).flatten()
        self.pop_distribution = item_popularity / item_popularity.sum()

        if self.split_set == 'val':
            val_lhs = pd.read_csv(os.path.join(self.data_path, 'listening_history_val.csv'))

            val_csr = sp.csr_matrix(
                (np.ones(len(val_lhs), dtype=np.int16), (val_lhs.user_id, val_lhs.item_id)),
                shape=(self.n_users, self.n_items))

            val_coo = sp.coo_matrix(val_csr)

            self.iteration_matrix = val_coo
            self.sampling_matrix = val_csr + train_csr

        elif self.split_set == 'test':
            test_lhs = pd.read_csv(os.path.join(self.data_path, 'listening_history_test.csv'))

            test_csr = sp.csr_matrix(
                (np.ones(len(test_lhs), dtype=np.int16), (test_lhs.user_id, test_lhs.item_id)),
                shape=(self.n_users, self.n_items))

            test_coo = sp.coo_matrix(test_csr)

            self.iteration_matrix = test_coo
            self.sampling_matrix = test_csr + train_csr

        elif self.split_set == 'train':
            train_coo = sp.coo_matrix(train_csr)

            self.iteration_matrix = train_coo
            self.sampling_matrix = train_csr

    def _neg_sample_uniform(self, row_idx: int) -> np.array:
        """
        For a specific user, it samples n_neg items u.a.r.
        :param row_idx: user id (or row in the matrix)
        :return: npy array containing the negatively sampled items.
        """

        consumed_items = self.sampling_matrix.indices[
                         self.sampling_matrix.indptr[row_idx]:self.sampling_matrix.indptr[row_idx + 1]]

        # Uniform distribution without items consumed by the user
        p = np.ones(self.n_items)
        p[consumed_items] = 0.  # Excluding consumed items
        p = p / p.sum()

        sampled = np.random.choice(np.arange(self.n_items), self.n_neg, replace=False, p=p)

        return sampled

    def _neg_sample_popular(self, row_idx: int) -> np.array:
        """
        For a specific user, it samples n_neg items considering the frequency of appearance of items in the dataset, i.e.
        p(i being neg) ∝ (pop_i)^0.75.
        :param row_idx: user id (or row in the matrix)
        :return: npy array containing the negatively sampled items.
        """
        consumed_items = self.sampling_matrix.indices[
                         self.sampling_matrix.indptr[row_idx]:self.sampling_matrix.indptr[row_idx + 1]]

        p = self.pop_distribution.copy()
        p[consumed_items] = 0.  # Excluding consumed items
        p = np.power(p, .75)  # Squashing factor alpha = .75
        p = p / p.sum()

        sampled = np.random.choice(np.arange(self.n_items), self.n_neg, replace=False, p=p)
        return sampled

    def __len__(self) -> int:
        return self.iteration_matrix.nnz

    def __getitem__(self, index) -> T_co:
        """
        Loads the (user,item) pair associated to the index and performs the negative sampling.
        :param index: (user,item) index pair (as defined by the COO.data vector)
        :return: (user_idx,item_idxs,labels) where
            user_idx: is the index of the user
            item_idxs: is a npy array containing the items indexes. The positive item is in the 1st position followed
                        by the negative items indexes. Shape is (1 + n_neg,)
            labels: npy array containing the labels. First position is 1, the others are 0. Shape is (1 + n_neg,).

        """

        user_idx = self.iteration_matrix.row[index].astype('int64')
        item_idx_pos = self.iteration_matrix.col[index]

        # Select the correct negative sampling strategy
        if self.neg_strategy == 'uniform':
            neg_samples = self._neg_sample_uniform(user_idx)
        elif self.neg_strategy == 'popular':
            neg_samples = self._neg_sample_popular(user_idx)
        else:
            raise ValueError(f'Negative Sampling Strategy <{self.neg_strategy}> not implemented ... Yet')

        item_idxs = np.concatenate(([item_idx_pos], neg_samples)).astype('int64')

        labels = np.zeros(1 + self.n_neg, dtype='float32')
        labels[0] = 1.

        return user_idx, item_idxs, labels


class UserRecDataset(RecDataset):

    def __init__(self, data_path: str, split_set: str, pos_strategy: str = 'n_pos', neg_strategy: str = 'ratio',
                 n_pos: int = 100, ro: int = None, n_neg: int = None):
        """
        This class iterates over the dataset user-wise. For each user, the class return n_pos positive instances and n_neg instances.
        """
        assert pos_strategy in ['n_pos', 'all'], f'Strategy for sampling positive ({pos_strategy}) not implemented'
        assert neg_strategy in ['n_neg', 'ratio',
                                'all'], f'Strategy for sampling negative ({neg_strategy}) not implemented'

        if pos_strategy == 'n_pos':
            assert n_pos >= 0, f'Value for n_pos ({n_pos}) not valid'
        if neg_strategy == 'n_neg':
            assert n_neg >= 0, f'Value for n_neg ({n_neg}) not valid'
        elif neg_strategy == 'ratio':
            assert ro is not None and ro >= 0, f'Value for ro ({ro}) not valid'

        super().__init__(data_path, split_set, 0)  # todo: tbh this only is used to load the data

        self.pos_strategy = pos_strategy
        self.neg_strategy = neg_strategy
        self.n_pos = n_pos
        self.ro = ro
        self.n_neg = n_neg

        self.iteration_matrix = sp.csr_matrix(self.iteration_matrix)

    def __len__(self):
        return self.n_users

    def __getitem__(self, user_index) -> T_co:

        # Sampling positives
        user_include_items = self.iteration_matrix[user_index].indices

        if self.pos_strategy == 'all' or (self.pos_strategy == 'n_pos' and self.n_pos >= len(user_include_items)):
            pos_items = user_include_items.copy()
            np.random.shuffle(pos_items)
        else:
            pos_items = np.random.choice(user_include_items, self.n_pos, replace=False)

        n_pos_items = len(pos_items)

        # Sampling negatives
        mask_neg = np.ones(self.n_items, dtype=bool)
        user_exclude_items = self.sampling_matrix[user_index].indices
        mask_neg[user_exclude_items] = False
        all_negative_items = np.arange(self.n_items)[mask_neg]

        n_items_left = len(all_negative_items)

        if self.neg_strategy == 'all' or (self.neg_strategy == 'n_neg' and self.n_neg >= n_items_left) or (
                self.neg_strategy == 'ratio' and n_pos_items * self.ro >= n_items_left):
            # Take all the negatives
            neg_items = all_negative_items.copy()
            np.random.shuffle(neg_items)
        elif self.neg_strategy == 'n_neg':
            neg_items = np.random.choice(all_negative_items, self.n_neg, replace=False)
        else:
            neg_items = np.random.choice(all_negative_items, n_pos_items * self.ro, replace=False)

        # Concatenating

        items = np.concatenate([pos_items, neg_items])

        # Labels
        pos_labels = np.ones_like(pos_items)
        neg_labels = np.zeros_like(neg_items)
        labels = np.concatenate([pos_labels, neg_labels])

        return user_index, items, labels


def get_recdataset_dataloader(data_path: str, split_set: str, n_neg: int, neg_strategy='uniform',
                              **loader_params) -> data.DataLoader:
    """
    Returns the dataloader for a RecDataset
    :param data_path, ... ,neg_strategy: check RecDataset class for info about these parameters
    :param loader_params: parameters for the Dataloader
    :return:
    """
    recdataset = RecDataset(data_path, split_set, n_neg, neg_strategy)

    return data.DataLoader(recdataset, **loader_params)


def get_userrecdataset_dataloader(data_path: str, split_set: str, pos_strategy: str = 'n_pos',
                                  neg_strategy: str = 'ratio',
                                  n_pos: int = 100, ro: int = None, n_neg: int = None,
                                  **loader_params) -> data.DataLoader:
    """
    Returns the dataloader for a UserRecDataset
    :param data_path, ... ,n_neg: check UserRecDataset class for info about these parameters
    :param loader_params: parameters for the Dataloader
    :return:
    """
    userrecdataset = UserRecDataset(data_path, split_set, pos_strategy, neg_strategy, n_pos, ro, n_neg)

    return data.DataLoader(userrecdataset, **loader_params)
