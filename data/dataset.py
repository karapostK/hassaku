import os

import numpy as np
import pandas as pd
from scipy import sparse as sp
from torch.utils import data

"""
The following classes are used to supply the recommender system data to the different methods. In 'data_path', there 
should be the following csv files:
- user_idxs.csv: containing at least the column `user_idx` which is the row index of the user in the interaction matrix.
        Possibly, the file also contains the 'id' used in the original dataset. Used in Train and Eval.
- item_idxs.csv: containing at least the column `item_idx` which is the column index of the item in the interaction matrix.
        Possibly, the file also contains the 'id' used in the original dataset. Used in Train and Eval.
- listening_history_train.csv: containing at least the columns `user_idx` and `item_idx` which corresponds to the entries
        in the interaction matrix used for training. Additional columns are allowed. Used in Train.
- listening_history_val.csv: same as listening_history_train.csv but contains the data used for validation. Used in
        Eval when split_set == val.
- listening_history_test.csv: same as listening_history_train.csv but contains the data used for test. Used in Eval 
        when split_set == test.
"""


class RecDataset(data.Dataset):
    """
    Dataset to hold Recommender System data and train collaborative filtering algorithms. It allows iteration over the
    dataset of positive interactions.
    """

    def __init__(self, data_path: str, split_set: str):
        """
        :param data_path: Path to the directory with listening_history_train.csv, user_idxs.csv, item_idxs.csv
        """
        assert split_set in ['train', 'val', 'test'], f'<{split_set}> is not a valid value for split set!'
        self.data_path = data_path
        self.split_set = split_set

        self.n_users = None
        self.n_items = None

        self.iteration_matrix = None

        self.pop_distribution = None

        self.load_data()

        self.name = "RecDataset"
        print(f'Built {self.name} module \n'
              f'- data_path: {self.data_path} \n'
              f'- split_set: {self.split_set} \n'
              f'- n_users: {self.n_users} \n'
              f'- n_items: {self.n_items} \n'
              f'- n_interactions: {self.iteration_matrix.nnz} \n')

    def load_data(self):
        print('Loading data')

        user_idxs = pd.read_csv(os.path.join(self.data_path, 'user_idxs.csv'))
        item_idxs = pd.read_csv(os.path.join(self.data_path, 'item_idxs.csv'))

        self.n_users = len(user_idxs)
        self.n_items = len(item_idxs)

        lhs = pd.read_csv(os.path.join(self.data_path, f'listening_history_{self.split_set}.csv'))

        if self.split_set == 'train':
            type_matrix = sp.coo_matrix  # During training we iterate over the single interactions
        else:
            type_matrix = sp.csr_matrix  # During evaluation we iterate user-wise

        self.iteration_matrix = type_matrix(
            (np.ones(len(lhs), dtype=np.int16), (lhs.user_idx, lhs.item_idx)),
            shape=(self.n_users, self.n_items))

        # Computing the popularity distribution
        item_popularity = np.array(self.iteration_matrix.sum(axis=0)).flatten()
        self.pop_distribution = item_popularity / item_popularity.sum()

        print('End loading data')

    def __len__(self):
        return self.iteration_matrix.nnz

    def __getitem__(self, index):
        user_idx = self.iteration_matrix.row[index].astype('int64')
        item_idx = self.iteration_matrix.col[index].astype('int64')

        return user_idx, item_idx, 1.


class TrainRecDataset(RecDataset):
    """
    Dataset to hold Recommender System data and train collaborative filtering algorithms. It allows iteration over the
    dataset of positive interaction and offers negative sampling functionalities (e.g. for each positive pair,
    sample n_neg negative samples).

    Additional notes:
    The data is loaded twice. Once the data is stored in a COO matrix to easily iterate over the dataset. Once in a CSR
    matrix to carry out fast negative sampling with the slicing functionalities.
    """

    def __init__(self, data_path: str, n_neg: int = 10, neg_sampling_strategy: str = 'uniform',
                 squashing_factor_pop_sampling: float = 0.75):
        """
        :param data_path: Path to the directory with listening_history_train.csv, user_idxs.csv, item_idxs.csv
        :param n_neg: Number of negative samples to take for each positive interaction
        :param neg_sampling_strategy: Either 'uniform' or 'popular'. See the respective functions for more details.
        :param squashing_factor_pop_sampling: Squashing factor for the popularity sampling. Ignored if neg_sampling_strategy = 'uniform'
        """

        super().__init__(data_path, 'train')
        assert neg_sampling_strategy in ['uniform', 'popular'], \
            f'<{neg_sampling_strategy}> is not a valid negative sampling strategy!'
        assert squashing_factor_pop_sampling >= 0, 'Squashing factor for popularity sampling should be positive!'

        self.n_neg = n_neg
        self.neg_sampling_strategy = neg_sampling_strategy
        self.squashing_factor_pop_sampling = squashing_factor_pop_sampling

        self.sampling_matrix = sp.csr_matrix(self.iteration_matrix)

        if self.neg_sampling_strategy == 'uniform':
            self.neg_sampler = self._neg_sample_uniform
        elif self.neg_sampling_strategy == 'popular':
            self.neg_sampler = self._neg_sample_popular
        else:
            raise ValueError('Error with the choice of neg_sampling strategy')

        self.name = 'TrainRecDataset'
        print(f'Built {self.name} module \n'
              f'- n_neg: {self.n_neg} \n'
              f'- neg_sampling_strategy: {self.neg_sampling_strategy} \n'
              f'- squashing_factor_pop_sampling: {self.squashing_factor_pop_sampling} \n')

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

    def _neg_sample_popular(self, row_idx: int, n_neg: int) -> np.array:
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
        p = np.power(p, self.squashing_factor_pop_sampling)  # Applying squashing factor alpha
        p = p / p.sum()

        sampled = np.random.choice(np.arange(self.n_items), n_neg, replace=False, p=p)
        return sampled

    def __getitem__(self, index):

        user_idx, item_idx_pos, _ = super().__getitem__(index)

        # Negative sampling
        neg_samples = self.neg_sampler(user_idx, self.n_neg)

        item_idxs = np.concatenate(([item_idx_pos], neg_samples)).astype('int64')

        labels = np.zeros(1 + self.n_neg, dtype='float32')
        labels[0] = 1.

        return user_idx, item_idxs, labels


class FullEvalDataset(RecDataset):
    """
    Dataset to hold Recommender System data and evaluate collaborative filtering algorithms. It allows iteration over
    all the users and compute the scores for all items (FullEvaluation).
    """

    def __init__(self, data_path: str, split_set: str, avoid_zeros_users: bool = True):
        """
        :param data_path: Path to the directory with listening_history_{val,test}.csv, user_idxs.csv, item_idxs.csv
        :param split_set: Either 'val' or 'test'
        :param avoid_zeros_users: Whether the dataset will also return users with no items on val/test data
        """

        super().__init__(data_path, split_set)

        self.avoid_zeros_users = avoid_zeros_users

        self.idx_to_user = None
        if self.avoid_zeros_users:
            items_per_users = self.iteration_matrix.getnnz(axis=-1)
            self.idx_to_user = np.where(items_per_users > 0)[0]

        self.load_data()

        self.name = 'FullEvalDataset'

        print(f'Built {self.name} module \n'
              f'- avoid_zeros_users: {self.avoid_zeros_users} \n')

    def __len__(self):
        if self.avoid_zeros_users:
            return len(self.idx_to_user)
        else:
            return self.n_users

    def __getitem__(self, user_index):
        if self.avoid_zeros_users:
            user_index = self.idx_to_user[user_index]
        return user_index, np.arange(self.n_items), self.iteration_matrix[user_index].toarray().squeeze().astype(
            'float32')


class StubDataset(data.Dataset):
    """
    Dataset used mostly for testing purposes
    """

    def __init__(self, n_users: int = 4000, n_items: int = 80000):
        self.n_users = n_users
        self.n_items = n_items

        self.evaluation_matrix = sp.lil_matrix((self.n_users, self.n_items))

        for user in range(self.n_users):
            items = np.random.randint(0, self.n_items, 20)
            self.evaluation_matrix[user, items] = 1
        self.evaluation_matrix = sp.csr_matrix(self.evaluation_matrix)

    def __len__(self):
        return self.n_users

    def __getitem__(self, user_index):
        return user_index, np.arange(self.n_items), self.evaluation_matrix[user_index].toarray().squeeze().astype(
            'float32')
