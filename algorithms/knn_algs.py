import os
from abc import ABC
from functools import partial

import numpy as np
from scipy import sparse as sp

from algorithms.base_classes import SparseMatrixBasedRecommenderAlgorithm
from utilities.similarities import SimilarityFunctionEnum


class KNNAlgorithm(SparseMatrixBasedRecommenderAlgorithm, ABC):

    def __init__(self, sim_func_enum: SimilarityFunctionEnum = SimilarityFunctionEnum.cosine, k: int = 100, **kwargs):
        """
        Abstract class for K-nearest neighbours
        :param sim_func_enum: similarity function to use
        :param k: number of k nearest neighbours to consider
        :param kwargs: additional parameters for the similarity function (e.g. alpha for asymmetric cosine)
        """
        super().__init__()

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
              f'- sim_func: {self.sim_func.__name__} \n'
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

        sim_mtx = take_only_top_k(self.sim_func(matrix), k=self.k)

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

        sim_mtx = take_only_top_k(self.sim_func(matrix.T), k=self.k)

        self.pred_mtx = matrix @ sim_mtx.T

        print('End Fitting')


def take_only_top_k(sim_mtx: sp.csr_matrix, k=100):
    """
    It slims down the similarity matrix by only picking the top-k most similar users/items for each user/item.
    This also allow to perform the prediction with a simple matrix multiplication
    """

    new_data = []
    new_indices = []
    new_indptr = [0]

    n_entities = sim_mtx.shape[0]

    cumulative_sum = 0

    for idx in range(n_entities):
        start_idx = sim_mtx.indptr[idx]
        end_idx = sim_mtx.indptr[idx + 1]

        data = sim_mtx.data[start_idx:end_idx]
        ind = sim_mtx.indices[start_idx:end_idx]

        # Avoiding taking the user/item itself
        if len(data) > 0:
            # The if is there to avoid cases where there are no closest neighbours (so even the sim to itself is 0)
            self_idx = np.where(ind == idx)[0][0]
            data[self_idx] = 0.

        top_k_indxs = np.argsort(-data)[:k]

        top_k_data = data[top_k_indxs]
        top_k_indices = ind[top_k_indxs]

        new_data += list(top_k_data)
        new_indices += list(top_k_indices)
        cumulative_sum += len(top_k_data)
        new_indptr.append(cumulative_sum)

    return sp.csr_matrix((new_data, new_indices, new_indptr), shape=sim_mtx.shape)
