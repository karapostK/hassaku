import os
from abc import ABC
from functools import partial

import numpy as np
from scipy import sparse as sp

from algorithms.base_classes import SparseMatrixBasedRecommenderAlgorithm
from utilities.similarities import SimilarityFunctionEnum, compute_similarity_top_k


class KNNAlgorithm(SparseMatrixBasedRecommenderAlgorithm, ABC):

    def __init__(self, sim_func_enum: SimilarityFunctionEnum = SimilarityFunctionEnum.cosine, k: int = 100, **kwargs):
        """
        Abstract class for K-nearest neighbours
        :param sim_func_enum: similarity function to use
        :param k: number of k nearest neighbours to consider
        :param kwargs: additional parameters for the similarity function (e.g. alpha for asymmetric cosine)
        """
        super().__init__()

        self.BLOCK_SIZE = 10000  # how many max entities are involved in the similarity computation at one point in time.

        self.sim_func_enum = sim_func_enum
        self.sim_func = sim_func_enum.value

        if self.sim_func_enum == SimilarityFunctionEnum.asymmetric_cosine:
            self.sim_func = partial(self.sim_func, kwargs['alpha'])
        elif self.sim_func_enum == SimilarityFunctionEnum.tversky:
            self.sim_func = partial(self.sim_func, kwargs['alpha'], kwargs['beta'])

        self.k = k

        self.pred_mtx = None

        self.name = 'KNNAlgorithm'

        print(f'Built {self.name} module \n'
              f'- sim_func: {self.sim_func_enum.name} \n'
              f'- k: {self.k} \n')

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
        sim_func_params = conf['sim_func_params']
        k = conf['k']
        sim_func = SimilarityFunctionEnum[sim_func_params['sim_func_name']]
        alpha = sim_func_params['alpha'] if 'alpha' in sim_func_params else None
        beta = sim_func_params['beta'] if 'beta' in sim_func_params else None
        if conf['alg'] == 'uknn':
            return UserKNN(sim_func, k, alpha=alpha, beta=beta)
        else:
            return ItemKNN(sim_func, k, alpha=alpha, beta=beta)


class UserKNN(KNNAlgorithm):

    def __init__(self, sim_func: SimilarityFunctionEnum = SimilarityFunctionEnum.cosine, k: int = 100, **kwargs):
        super().__init__(sim_func, k, **kwargs)
        self.name = 'UserKNN'
        print(f'Built {self.name} module \n')

    def fit(self, matrix: sp.spmatrix):
        """
        :param matrix: user x item sparse matrix
        """
        print('Starting Fitting')

        sim_mtx = compute_similarity_top_k(matrix, self.sim_func, self.k, self.BLOCK_SIZE)

        self.pred_mtx = sim_mtx @ matrix
        print('End Fitting')


class ItemKNN(KNNAlgorithm):

    def __init__(self, sim_func: SimilarityFunctionEnum = SimilarityFunctionEnum.cosine, k: int = 100, **kwargs):
        super().__init__(sim_func, k, **kwargs)
        self.name = 'ItemKNN'
        print(f'Built {self.name} module \n')

    def fit(self, matrix: sp.spmatrix):
        """
        :param matrix: user x item sparse matrix
        """
        print('Starting Fitting')

        sim_mtx = compute_similarity_top_k(matrix.T, self.sim_func, self.k, self.BLOCK_SIZE)

        self.pred_mtx = matrix @ sim_mtx.T

        print('End Fitting')
