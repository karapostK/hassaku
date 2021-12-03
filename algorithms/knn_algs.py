import typing
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import torch
from scipy import sparse as sp

from base_classes import RecommenderAlgorithm
from utilities.similarities import SimilarityFunction


class KNNAlgorithm(RecommenderAlgorithm, ABC):

    def __init__(self, sim_func: SimilarityFunction = SimilarityFunction.cosine, k: int = 100, **kwargs):
        """
        Abstract class for K-nearest neighbours
        :param sim_func: similarity function to use
        :param k: number of k nearest neighbours to consider
        :param kwargs: additional parameters for the similarity function (e.g. alpha for asymmetric cosine)
        """
        super().__init__()

        self.sim_func = sim_func
        if self.sim_func == SimilarityFunction.asymmetric_cosine:
            self.sim_func = partial(self.sim_func, kwargs['alpha'])
        elif self.sim_func == SimilarityFunction.tversky:
            self.sim_func = partial(self.sim_func, kwargs['alpha'], kwargs['beta'])

        self.k = k

        self.pred_mtx = None

        self.name = 'KNNAlgorithm'

        print(f'Built {self.name} module \n'
              f'- sim_func: {self.sim_func} \n'
              f'- k: {self.k} \n')

    @abstractmethod
    def fit(self, matrix: sp.spmatrix):
        """
        :param matrix: user x item sparse matrix
        """
        pass

    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> typing.Union:
        assert self.pred_mtx is not None, 'Prediction Matrix not computed, run fit!'
        out = self.pred_mtx[u_idxs[:, None], i_idxs]
        return out


class UserKNN(KNNAlgorithm):

    def __init__(self, sim_func: SimilarityFunction = SimilarityFunction.cosine, k: int = 100, **kwargs):
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
        self.pred_mtx = self.pred_mtx.toarray()  # Not elegant but materializing the whole matrix makes the rest of the code easier to write/read
        print('End Fitting')


class ItemKNN(KNNAlgorithm):

    def __init__(self, sim_func: SimilarityFunction = SimilarityFunction.cosine, k: int = 100, **kwargs):
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
        self.pred_mtx = self.pred_mtx.toarray()  # Not elegant but materializing the whole matrix makes the rest of the code easier to write/read
        print('End Fitting')


def take_only_top_k(sim_mtx, k=100):
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
