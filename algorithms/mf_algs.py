import typing

import numpy as np
import torch
from scipy import sparse as sp
from scipy.sparse.linalg import svds

from base_classes import RecommenderAlgorithm


class SVDAlgorithm(RecommenderAlgorithm):

    def __init__(self, factors: int = 100):
        """
        Implement the Singular Value Decomposition.
        :param factors: number of latent factors.
        """
        super().__init__()
        self.factors = factors

        self.users_factors = None
        self.items_factors = None

        self.name = 'SVDAlgorithm'

        print('SVDAlgorithm class created')

    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> typing.Union[np.ndarray, torch.Tensor]:
        assert (self.users_factors is not None) and \
               (self.items_factors is not None), 'User and Item factors are None! Call fit before predict'

        batch_users = self.users_factors[u_idxs]
        batch_items = self.items_factors[i_idxs]
        out = (batch_items * batch_users[:, None, :]).sum(axis=-1)  # Carrying out the dot product

        return out

    def fit(self, matrix: sp.csr_matrix):
        matrix = matrix.asfptype()  # casting to float

        u, s, vt = svds(matrix, k=self.factors)

        self.users_factors = u * s
        self.items_factors = vt.T
