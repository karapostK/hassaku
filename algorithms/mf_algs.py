import typing

import implicit
import numpy as np
import torch
from scipy import sparse as sp
from scipy.sparse.linalg import svds

from algorithms.base_classes import RecommenderAlgorithm


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

        print(f'Built {self.name} module \n'
              f'- factors: {self.factors} ')

    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> typing.Union[np.ndarray, torch.Tensor]:
        assert (self.users_factors is not None) and \
               (self.items_factors is not None), 'User and Item factors are None! Call fit before predict'

        batch_users = self.users_factors[u_idxs]
        batch_items = self.items_factors[i_idxs]
        out = (batch_items * batch_users[:, None, :]).sum(axis=-1)  # Carrying out the dot product

        return out

    def fit(self, matrix: sp.spmatrix):
        print('Starting Fitting')
        matrix = matrix.asfptype()  # casting to float

        u, s, vt = svds(matrix, k=self.factors)

        self.users_factors = u * s
        self.items_factors = vt.T
        print('End Fitting')

    def save_model_to_path(self, path: str):
        np.savez(path, users_factors=self.users_factors, items_factors=self.items_factors)

    def load_model_from_path(self, path: str):
        with np.load(path) as array_dict:
            self.users_factors = array_dict['users_factors']
            self.items_factors = array_dict['items_factors']


class AlternatingLeastSquare(RecommenderAlgorithm):
    def __init__(self, alpha: int, factors: int, regularization: float, n_iterations: int):
        super().__init__()
        '''
        From Collaborative Filtering for Implicit Datasets (http://yifanhu.net/PUB/cf.pdf)
        :param alpha: controls the confidence value (see original paper Collaborative Filtering for Implicit Datasets)
        :param factors: embedding size
        :param regularization: regularization factor (the l2 factor)
        :param iter: number of iterations for ALS
        '''

        self.alpha = alpha
        self.factors = factors
        self.regularization = regularization
        self.n_iterations = n_iterations

        self.users_factors = None
        self.items_factors = None

        self.name = "AlternatingLeastSquare"

        print(f'Built {self.name} module \n'
              f'- alpha: {self.alpha} \n'
              f'- factors: {self.factors} \n'
              f'- regularization: {self.regularization} \n'
              f'- n_iterations: {self.n_iterations} \n')

    def fit(self, matrix: sp.spmatrix, use_gpu: bool = True):
        print('Starting Fitting')

        matrix = sp.csr_matrix(matrix.T)
        als = implicit.als.AlternatingLeastSquares(factors=self.factors,
                                                   regularization=self.regularization,
                                                   iterations=self.n_iterations,
                                                   use_gpu=use_gpu,
                                                   num_threads=10)
        als.fit(self.alpha * matrix)

        self.items_factors = als.item_factors.to_numpy()
        self.users_factors = als.user_factors.to_numpy()
        print('End Fitting')

    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> typing.Union[np.ndarray, torch.Tensor]:
        assert (self.users_factors is not None) and \
               (self.items_factors is not None), 'User and Item factors are None! Call fit before predict'

        batch_users = self.users_factors[u_idxs]
        batch_items = self.items_factors[i_idxs]
        out = (batch_items * batch_users[:, None, :]).sum(axis=-1)  # Carrying out the dot product

        return out

    def save_model_to_path(self, path: str):
        np.savez(path, users_factors=self.users_factors, items_factors=self.items_factors)

    def load_model_from_path(self, path: str):
        with np.load(path) as array_dict:
            self.users_factors = array_dict['users_factors']
            self.items_factors = array_dict['items_factors']
