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
    Dataset to hold Recommender System data in the format of a pandas dataframe.
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

        self.lhs = None

        self._load_data()

        self.name = "RecDataset"
        print(f'Built {self.name} module \n'
              f'- data_path: {self.data_path} \n'
              f'- split_set: {self.split_set} \n'
              f'- n_users: {self.n_users} \n'
              f'- n_items: {self.n_items} \n'
              f'- n_interactions: {len(self.lhs)} \n')

    def _load_data(self):
        print('Loading data')

        user_idxs = pd.read_csv(os.path.join(self.data_path, 'user_idxs.csv'))
        item_idxs = pd.read_csv(os.path.join(self.data_path, 'item_idxs.csv'))

        self.n_users = len(user_idxs)
        self.n_items = len(item_idxs)

        self.lhs = self._load_lhs(self.split_set)

        print('End loading data')

    def _load_lhs(self, split_set: str):
        return pd.read_csv(os.path.join(self.data_path, f'listening_history_{split_set}.csv'))

    def __len__(self):
        raise NotImplementedError("RecDataset does not support __len__ or __getitem__. Please use TrainRecDataset for"
                                  "training or FullEvalDataset for evaluation.")

    def __getitem__(self, index):
        raise NotImplementedError("RecDataset does not support __len__ or __getitem__. Please use TrainRecDataset for"
                                  "training or FullEvalDataset for evaluation.")


class TrainRecDataset(RecDataset):
    """
    Dataset to hold Recommender System data and train collaborative filtering algorithms. It allows iteration over the
    dataset of positive interaction. It also stores the item popularity distribution over the training data.

    Additional notes:
    The data is loaded twice. Once the data is stored in a COO matrix to easily iterate over the dataset. Once in a CSR
    matrix to carry out fast negative sampling with the user-wise slicing functionalities (see also collate_fn in data/dataloader.py)
    """

    def __init__(self, data_path: str, delete_lhs: bool = True):
        """
        :param data_path: Path to the directory with listening_history_train.csv, user_idxs.csv, item_idxs.csv
        :param delete_lhs: Whether the pandas dataframe should be deleted after creating the iteration/sampling mtxs.
        """

        super().__init__(data_path, 'train')

        self.delete_lhs = delete_lhs

        self.iteration_matrix = None
        self.sampling_matrix = None

        self.pop_distribution = None

        self._prepare_data()

        self.name = 'TrainRecDataset'
        print(f'Built {self.name} module \n'
              f'- delete_lhs: {self.delete_lhs} \n')

    def _prepare_data(self):
        self.iteration_matrix = sp.coo_matrix(
            (np.ones(len(self.lhs), dtype=np.int16), (self.lhs.user_idx, self.lhs.item_idx)),
            shape=(self.n_users, self.n_items))

        self.sampling_matrix = sp.csr_matrix(self.iteration_matrix)

        item_popularity = np.array(self.iteration_matrix.sum(axis=0)).flatten()
        self.pop_distribution = item_popularity / item_popularity.sum()

        if self.delete_lhs:
            del self.lhs

    def __len__(self):
        return self.iteration_matrix.nnz

    def __getitem__(self, index):
        user_idx = self.iteration_matrix.row[index].astype('int64')
        item_idx = self.iteration_matrix.col[index].astype('int64')

        return user_idx, item_idx, 1.


class FullEvalDataset(RecDataset):
    """
    Dataset to hold Recommender System data and evaluate collaborative filtering algorithms. It allows iteration over
    all the users and compute the scores for all items (FullEvaluation). It also holds data from training and validation
    that needs to be excluded from the evaluation:
    During validation, items in the training data for a user are excluded as labels
    During test, items in the training data and validation for a user are excluded as labels
    """

    def __init__(self, data_path: str, split_set: str, delete_lhs: bool = True):
        """
        :param data_path: Path to the directory with listening_history_{val,test}.csv, user_idxs.csv, item_idxs.csv
        :param split_set: Either 'val' or 'test'
        :param delete_lhs: Whether the pandas dataframe should be deleted after creating the iteration/sampling mtxs.
        """

        super().__init__(data_path, split_set)

        self.delete_lhs = delete_lhs

        self.idx_to_user = None
        self.iteration_matrix = None
        self.exclude_data = None

        self._prepare_data()

        self.name = 'FullEvalDataset'

        print(f'Built {self.name} module \n'
              f'- delete_lhs: {self.delete_lhs} \n')

    def _prepare_data(self):
        self.iteration_matrix = sp.csr_matrix(
            (np.ones(len(self.lhs), dtype=np.int16), (self.lhs.user_idx, self.lhs.item_idx)),
            shape=(self.n_users, self.n_items))

        # Load Train data as well
        train_lhs = self._load_lhs('train')
        self.exclude_data = sp.csr_matrix(
            (np.ones(len(train_lhs), dtype=bool), (train_lhs.user_idx, train_lhs.item_idx)),
            shape=(self.n_users, self.n_items)
        )
        # If 'split_test' load also Valid data
        if self.split_set == 'test':
            val_lhs = self._load_lhs('val')
            self.exclude_data += sp.csr_matrix(
                (np.ones(len(val_lhs), dtype=bool), (val_lhs.user_idx, val_lhs.item_idx)),
                shape=(self.n_users, self.n_items)
            )

        if self.delete_lhs:
            del self.lhs

    def __len__(self):
        return self.n_users

    def __getitem__(self, user_index):
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
