import typing

import numpy as np
import torch
from scipy import sparse as sp

from base_classes import RecommenderAlgorithm


class P3alpha(RecommenderAlgorithm):
    """
    A simple random walk algorithm. https://dl.acm.org/doi/pdf/10.1145/2567948.2579244
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

        # Building Adjacency Matrix #todo maybe improve the construction

        A = sp.lil_matrix((total_nodes, total_nodes))
        A[:n_users, n_users:] = matrix
        A[n_users:, :n_users] = matrix.T
        A = sp.csr_matrix(A)

        # Building Diagonal Matrix # todo improve

        D = sp.lil_matrix(A.shape)
        D[np.diag_indices(D.shape[0])] = 1 / diagonal
        D = sp.csr_matrix(D)

        # Transition Matrix

        P = D @ A

        del D, A  # Cleaning

        # First Step

        P2 = P @ P

        print('Frist Step done')
        # Second Step

        P3 = P2 @ P
        print('Second Step done')

        del P2, P  # Cleaning before imploding the memory

        # Taking only the user x item matrix

        P3 = P3[:n_users, n_users:]

        # Raising to the power of alpha (element-wise)
        P3 = P3.power(self.alpha)

        self.pred_mtx = P3.toarray()

        print('End fitting')

    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> typing.Union:
        assert self.pred_mtx is not None, 'Prediction Matrix not computed, run fit!'
        out = self.pred_mtx[u_idxs[:, None], i_idxs]
        return out
