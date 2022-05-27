import os

import numpy as np
from scipy import sparse as sp

from algorithms.base_classes import SparseMatrixBasedRecommenderAlgorithm


class P3alpha(SparseMatrixBasedRecommenderAlgorithm):
    """
    A simple random walk algorithm.  https://dl.acm.org/doi/pdf/10.1145/2567948.2579244
    See also https://www.cs.yale.edu/homes/spielman/561/lect10-18.pdf or https://people.math.osu.edu/husen.1/teaching/571/random_walks.pdf
    """

    def __init__(self, alpha: float = 1.9):
        super().__init__()

        assert alpha >= 0, f"Alpha ({alpha}) has to be greater or equal than 0"
        self.alpha = alpha
        self.name = 'P3alpha'

        self.pred_mtx = None
        print(f'Built {self.name} module \n'
              f'- alpha: {self.alpha}')

    def fit(self, matrix: sp.spmatrix):
        """
        :param matrix: user x item matrix
        """
        print('Start fitting')

        n_users, n_items = matrix.shape[0], matrix.shape[1]
        total_nodes = n_users + n_items

        item_sum = np.array(matrix.sum(axis=0)).flatten()
        user_sum = np.array(matrix.sum(axis=1)).flatten()
        diagonal = np.concatenate([user_sum, item_sum])

        # Building Adjacency Matrix

        A = sp.lil_matrix((total_nodes, total_nodes))
        A[:n_users, n_users:] = matrix
        A[n_users:, :n_users] = matrix.T
        A = sp.csr_matrix(A)

        # Building Diagonal Matrix

        D = sp.lil_matrix(A.shape)
        D[np.diag_indices(D.shape[0])] = 1 / diagonal
        D = sp.csr_matrix(D)

        # Transition Matrix

        P = D @ A

        del D, A  # Cleaning

        # Three Steps
        P3 = P ** 3

        del P

        # Taking only the user x item matrix

        P3 = P3[:n_users, n_users:]

        # Raising to the power of alpha (element-wise)
        P3 = P3.power(self.alpha)

        self.pred_mtx = P3

        print('End fitting')

    def save_model_to_path(self, path: str):
        path = os.path.join(path, 'model.npz')
        np.savez(path, pred_mtx=self.pred_mtx)
        print('Model Saved')

    def load_model_from_path(self, path: str):
        path = os.path.join(path, 'model.npz')
        with np.load(path) as array_dict:
            self.pred_mtx = array_dict['pred_mtx']
        print('Model Loaded')

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        alpha = conf['alpha']
        return P3alpha(alpha=alpha)

# class LightGCN(SGDBasedRecommenderAlgorithm):
#     """
#     http://staff.ustc.edu.cn/~hexn/papers/sigir20-LightGCN.pdf
#     """
#
#     def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, n_layers: int = 5):
#         super().__init__()
#
#         self.n_users = n_users
#         self.n_items = n_items
#         self.embedding_dim = embedding_dim
#         self.n_layers = n_layers
#
#         self.user_embeddings = nn.Embedding(self.n_users, self.embedding_dim)
#         self.item_embeddings = nn.Embedding(self.n_items, self.embedding_dim)
#
#         self.apply(general_weight_init)
#
#         self.name = 'LightGCN'
#
#         print(f'Built {self.name} module\n'
#               f'- embedding_dim: {self.embedding_dim} \n'
#               f'- n_layers: {self.n_layers} \n')
#
#     def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
