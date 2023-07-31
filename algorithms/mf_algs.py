import logging
import os
import typing

import maxvolpy
import numpy as np
import torch
from implicit.als import AlternatingLeastSquares
from scipy import sparse as sp
from scipy.sparse.linalg import svds

from algorithms.base_classes import SparseMatrixBasedRecommenderAlgorithm


class SVDAlgorithm(SparseMatrixBasedRecommenderAlgorithm):

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

        logging.info(f'Built {self.name} module \n'
                     f'- factors: {self.factors} ')

    def fit(self, matrix: sp.spmatrix):
        print('Starting Fitting')
        matrix = matrix.asfptype()  # casting to float

        u, s, vt = svds(matrix, k=self.factors)

        self.users_factors = u * s
        self.items_factors = vt.T
        print('End Fitting')

    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor):
        assert (self.users_factors is not None) and \
               (self.items_factors is not None), 'User and Item factors are None! Call fit before predict'

        batch_users = self.users_factors[u_idxs]
        batch_items = self.items_factors[i_idxs]
        out = (batch_items * batch_users[:, None, :]).sum(axis=-1)  # Carrying out the dot product

        return out

    def save_model_to_path(self, path: str):
        path = os.path.join(path, 'model.npz')
        np.savez(path, users_factors=self.users_factors, items_factors=self.items_factors)
        print('Model Saved')

    def load_model_from_path(self, path: str):
        path = os.path.join(path, 'model.npz')
        with np.load(path) as array_dict:
            self.users_factors = array_dict['users_factors']
            self.items_factors = array_dict['items_factors']
        print('Model Loaded')

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        return SVDAlgorithm(conf['n_factors'])


class AlternatingLeastSquare(SparseMatrixBasedRecommenderAlgorithm):
    def __init__(self, alpha: int, factors: int, regularization: float, n_iterations: int, use_gpu: bool = True):
        super().__init__()
        '''
        From Collaborative Filtering for Implicit Datasets (http://yifanhu.net/PUB/cf.pdf)
        Implementation from the implicit library
        :param alpha: controls the confidence value (see original paper Collaborative Filtering for Implicit Datasets)
            P.S. This value are the weights given to the positive examples
        :param factors: embedding size
        :param regularization: regularization factor (the l2 factor)
        :param iter: number of iterations for ALS
        :param use_gpu: whether to use the gpu for training
        '''

        self.alpha = alpha
        self.factors = factors
        self.regularization = regularization
        self.n_iterations = n_iterations
        self.use_gpu = use_gpu

        self.users_factors = None
        self.items_factors = None

        self.name = "AlternatingLeastSquare"

        logging.info(f'Built {self.name} module \n'
                     f'- alpha: {self.alpha} \n'
                     f'- factors: {self.factors} \n'
                     f'- regularization: {self.regularization} \n'
                     f'- n_iterations: {self.n_iterations} \n'
                     f'- use_gpu: {self.use_gpu} \n')

    def fit(self, matrix: sp.spmatrix):
        print('Starting Fitting')

        matrix = sp.csr_matrix(matrix)
        als = AlternatingLeastSquares(factors=self.factors,
                                      alpha=self.alpha,
                                      regularization=self.regularization,
                                      iterations=self.n_iterations,
                                      use_gpu=self.use_gpu,
                                      num_threads=10)
        als.fit(matrix)

        self.items_factors = als.item_factors
        self.users_factors = als.user_factors
        print('End Fitting')

    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> typing.Union[np.ndarray, torch.Tensor]:
        assert (self.users_factors is not None) and \
               (self.items_factors is not None), 'User and Item factors are None! Call fit before predict'

        batch_users = self.users_factors[u_idxs]
        batch_items = self.items_factors[i_idxs]
        out = (batch_users[:, None, :] * batch_items).sum(axis=-1)  # Carrying out the dot product

        return out

    def save_model_to_path(self, path: str):
        path = os.path.join(path, 'model.npz')
        np.savez(path, users_factors=self.users_factors, items_factors=self.items_factors)
        print('Model Saved')

    def load_model_from_path(self, path: str):
        path = os.path.join(path, 'model.npz')
        with np.load(path) as array_dict:
            self.users_factors = array_dict['users_factors']
            self.items_factors = array_dict['items_factors']
        print('Model Loaded')

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        return AlternatingLeastSquare(conf['alpha'], conf['factors'], conf['regularization'], conf['n_iterations'],
                                      conf['use_gpu'])


class RBMF(SparseMatrixBasedRecommenderAlgorithm):

    def __init__(self, n_representatives: int, lam: float = 1e-2):
        super().__init__()
        '''
        User Representative-based Matrix Factorization (from https://dl.acm.org/doi/10.1145/2043932.2043943)
        This implementation considers the basic RBMF, as proposed in the paper.
        :param n_representatives: number of representatives to pick
        :param lam: l2 regularization
        '''

        self.n_representatives = n_representatives
        self.lam = lam

        self.X = None
        self.C = None

        self.name = "RBMF"

        logging.info(f'Built {self.name} module \n'
                     f'- n_representatives: {self.n_representatives} \n'
                     f'- lam: {self.lam} \n')

    def fit(self, matrix: sp.spmatrix):
        print('Starting Fitting')
        matrix = sp.csr_matrix(matrix)
        matrix = matrix.asfptype()  # casting to float

        # Representative Pursuit
        u, _, _ = svds(matrix, k=self.n_representatives)  # Dimension Reduction
        indxs, _ = maxvolpy.maxvol.maxvol(u)  # Basis Selection (indexes of the users that maxime the square volume)
        C = matrix[indxs]  # Extracting rows form the original matrix [n_representatives, n_items]

        Inv = np.linalg.inv(C @ C.T + self.lam * np.eye(self.n_representatives))
        X = matrix @ C.T @ Inv

        self.X = np.array(X)  # [n_users, n_representatives]
        self.C = C.toarray().T  # [n_items, n_representatives]

        print('End Fitting')

    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> typing.Union[np.ndarray, torch.Tensor]:
        assert self.X is not None and self.C is not None, "X and C are none!"

        batch_users = self.X[u_idxs]
        batch_items = self.C[i_idxs]
        out = (batch_users[:, None, :] * batch_items).sum(axis=-1)  # Carrying out the dot product

        return out

    def save_model_to_path(self, path: str):
        path = os.path.join(path, 'model.npz')
        np.savez(path, X=self.X, C=self.C)
        print('Model Saved')

    def load_model_from_path(self, path: str):
        path = os.path.join(path, 'model.npz')
        with np.load(path) as array_dict:
            self.X = array_dict['X']
            self.C = array_dict['C']
        print('Model Loaded')

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        return RBMF(conf['n_representatives'], conf['lam'])
