import os

import numpy as np
import pandas as pd
from scipy import sparse as sp
from torch.utils import data

"""
The following classes are used to supply the recommender system data to the different methods. In 'data_path', there 
should be the following csv files:
- user_ids.csv: containing at least the column `user_id` which is the row index of the user in the interaction matrix.
        Possibly, the file also contains the 'id' used in the original dataset. Used in Train and Eval.
- item_ids.csv: containing at least the column `item_id` which is the column index of the item in the interaction matrix.
        Possibly, the file also contains the 'id' used in the original dataset. Used in Train and Eval.
- listening_history_train.csv: containing at least the columns `user_id` and `item_id` which corresponds to the entries
        in the interaction matrix used for training. Additional columns are allowed. Used in Train.
- listening_history_val.csv: same as listening_history_train.csv but contains the data used for validation. Used in
        Eval when split_set == val.
- listening_history_test.csv: same as listening_history_train.csv but contains the data used for test. Used in Eval 
        when split_set == test.
"""


class TrainRecDataset(data.Dataset):
    """
    Dataset to hold Recommender System data and train collaborative filtering algorithms. It allows iteration over the
    dataset of positive interaction and offers negative sampling functionalities (e.g. for each positive pair,
    sample n_neg negative samples).

    Additional notes:
    The data is loaded twice. Once the data is stored in a COO matrix to easily iterate over the dataset. Once in a CSR
    matrix to carry out fast negative sampling with the slicing functionalities.
    """

    def __init__(self, data_path: str, n_neg: int = 10, neg_sampling_strategy: str = 'uniform'):
        """
        :param data_path: Path to the directory with listening_history_train.csv, user_ids.csv, item_ids.csv
        :param n_neg: Number of negative samples to take for each positive interaction
        :param neg_sampling_strategy: Either 'uniform' or 'popular'. See the respective functions for more details.
        """

        assert neg_sampling_strategy in ['uniform', 'popular'], \
            f'<{neg_sampling_strategy}> is not a valid negative sampling strategy!'

        self.data_path = data_path
        self.n_neg = n_neg
        self.neg_sampling_strategy = neg_sampling_strategy

        self.n_users = None
        self.n_items = None

        self.iteration_matrix = None
        self.sampling_matrix = None

        if self.neg_sampling_strategy == 'uniform':
            self.neg_sampler = self._neg_sample_uniform
        elif self.neg_sampling_strategy == 'popular':
            self.neg_sampler = self._neg_sample_popular
        else:
            raise ValueError('Error with the choice of neg_sampling strategy')

        self.pop_distribution = None

        self.load_data()

        print(f'Built TrainRecDataset module \n'
              f'- data_path: {self.data_path} \n'
              f'- n_users: {self.n_users} \n'
              f'- n_items: {self.n_items} \n'
              f'- n_interactions: {self.iteration_matrix.nnz} \n'
              f'- n_neg: {self.n_neg} \n'
              f'- neg_sampling_strategy: {self.neg_sampling_strategy} \n')

    def load_data(self):
        print('Loading data')

        user_ids = pd.read_csv(os.path.join(self.data_path, 'user_ids.csv'))
        item_ids = pd.read_csv(os.path.join(self.data_path, 'item_ids.csv'))

        self.n_users = len(user_ids)
        self.n_items = len(item_ids)

        train_lhs = pd.read_csv(os.path.join(self.data_path, 'listening_history_train.csv'))

        self.sampling_matrix = sp.csr_matrix(
            (np.ones(len(train_lhs), dtype=np.int16), (train_lhs.user_id, train_lhs.item_id)),
            shape=(self.n_users, self.n_items))

        # Computing the popularity distribution (see _neg_sample_popular)
        item_popularity = np.array(self.sampling_matrix.sum(axis=0)).flatten()
        self.pop_distribution = item_popularity / item_popularity.sum()

        self.iteration_matrix = sp.coo_matrix(self.sampling_matrix)

        print('End loading data')

    def _neg_sample_uniform(self, row_idx: int, n_neg: int) -> np.array:
        """
        For a specific user, it samples n_neg items u.a.r.
        :param row_idx: user id (or row in the matrix)
        :param n_neg: number of negative samples
        :return: npy array containing the negatively sampled items.
        """

        consumed_items = self.sampling_matrix[row_idx].indices

        # Uniform distribution without items consumed by the user
        p = np.ones(self.n_items)
        p[consumed_items] = 0.  # Excluding consumed items
        p = p / p.sum()

        sampled = np.random.choice(np.arange(self.n_items), n_neg, replace=False, p=p)

        return sampled

    def _neg_sample_popular(self, row_idx: int, n_neg: int, squashing_factor=.75) -> np.array:
        """
        For a specific user, it samples n_neg items considering the frequency of appearance of items in the dataset, i.e.
        p(i being neg) ∝ (pop_i)^0.75.
        :param row_idx: user id (or row in the matrix)
        :param n_neg: number of negative samples
        :return: npy array containing the negatively sampled items.
        """
        consumed_items = self.sampling_matrix[row_idx].indices

        p = self.pop_distribution.copy()
        p[consumed_items] = 0.  # Excluding consumed items
        p = np.power(p, squashing_factor)  # Applying squashing factor alpha
        p = p / p.sum()

        sampled = np.random.choice(np.arange(self.n_items), n_neg, replace=False, p=p)
        return sampled

    def __len__(self):
        return self.iteration_matrix.nnz

    def __getitem__(self, index):

        user_idx = self.iteration_matrix.row[index].astype('int64')
        item_idx_pos = self.iteration_matrix.col[index]

        # Negative sampling
        neg_samples = self.neg_sampler(user_idx, self.n_neg)

        item_idxs = np.concatenate(([item_idx_pos], neg_samples)).astype('int64')

        labels = np.zeros(1 + self.n_neg, dtype='float32')
        labels[0] = 1.

        return user_idx, item_idxs, labels


class FullEvalDataset(data.Dataset):
    """
    Dataset to hold Recommender System data and evaluate collaborative filtering algorithms. It allows iteration over
    all the users and compute the scores for all items (FullEvaluation).
    """

    def __init__(self, data_path: str, split_set: str):
        """
        :param data_path: Path to the directory with listening_history_{val,test}.csv, user_ids.csv, item_ids.csv
        :param split_set: Either 'val' or 'test'
        """
        assert split_set in ['val', 'test'], f'<{split_set}> is not a valid value for split set!'

        self.data_path = data_path
        self.split_set = split_set

        self.n_users = None
        self.n_items = None

        self.evaluation_matrix = None

        self.load_data()

        print(f'Built FullEvalDataset module \n'
              f'- data_path: {self.data_path} \n'
              f'- n_users: {self.n_users} \n'
              f'- n_items: {self.n_items} \n'
              f'- n_interactions: {self.evaluation_matrix.nnz} \n'
              f'- split_set: {self.split_set} \n')

    def load_data(self):
        print('Loading data')

        user_ids = pd.read_csv(os.path.join(self.data_path, 'user_ids.csv'))
        item_ids = pd.read_csv(os.path.join(self.data_path, 'item_ids.csv'))

        self.n_users = len(user_ids)
        self.n_items = len(item_ids)

        eval_lhs = pd.read_csv(os.path.join(self.data_path, f'listening_history_{self.split_set}.csv'))

        self.evaluation_matrix = sp.csr_matrix(
            (np.ones(len(eval_lhs), dtype=np.int16), (eval_lhs.user_id, eval_lhs.item_id)),
            shape=(self.n_users, self.n_items))

        print('End loading data')

    def __len__(self):
        return self.n_users

    def __getitem__(self, user_index):
        return user_index, np.arange(self.n_items), self.evaluation_matrix[user_index].toarray().squeeze().astype('float32')
