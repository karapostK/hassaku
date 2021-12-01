import typing
from functools import partial

import numpy as np
import torch
from scipy import sparse as sp

from base_classes import RecommenderAlgorithm
from utilities.similarities import SimilarityFunction, sim_mapping_to_func


class UserKNN(RecommenderAlgorithm):

    def __init__(self, sim_func_choice: SimilarityFunction = SimilarityFunction.cosine):
        """
        Implement User k-nearest neighbours algorithms
        :param sim_func_choice: similarity function to use
        """
        super().__init__()
        self.sim_func_choice = sim_func_choice

        self.pred_mtx = None

        self.name = 'UserKNN'

        print('UserKNN class created')

    def fit(self, matrix: sp.spmatrix, k: int = 100, **kwargs):
        """
        :param matrix: user x item sparse matrix
        :param k: number of k nearest neighbours to consider
        :param kwargs: additional parameters for the similarity function (e.g. alpha for asymmetric cosine)
        """
        sim_fun = sim_mapping_to_func[self.sim_func_choice]
        if self.sim_func_choice == SimilarityFunction.asymmetric_cosine:
            sim_fun = partial(sim_fun, kwargs['alpha'])
        elif self.sim_func_choice == SimilarityFunction.tversky:
            sim_fun = partial(sim_fun, kwargs['alpha'], kwargs['beta'])

        sim_mtx = take_only_top_k(sim_fun(matrix), k=k)

        self.pred_mtx = sim_mtx @ matrix
        self.pred_mtx = self.pred_mtx.toarray()  # Not elegant but materializing the whole matrix makes the rest of the code easier to write/read

    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> typing.Union:
        out = self.pred_mtx[u_idxs[:, None], i_idxs]
        return out


class ItemKNN(RecommenderAlgorithm):

    def __init__(self, sim_func_choice: SimilarityFunction = SimilarityFunction.cosine):
        """
        Implement Item k-nearest neighbours algorithms
        :param sim_func_choice: similarity function to use
        """
        super().__init__()
        self.sim_func_choice = sim_func_choice

        self.pred_mtx = None

        self.name = 'ItemKNN'

        print('ItemKNN class created')

    def fit(self, matrix, k=100, **kwargs):
        """
        :param matrix: user x item matrix
        :param k: number of k nearest neighbours to consider
        :param kwargs: additional parameters for the similarity function (e.g. alpha for asymmetric cosine)
        """
        sim_fun = sim_mapping_to_func[self.sim_func_choice]
        if self.sim_func_choice == SimilarityFunction.asymmetric_cosine:
            sim_fun = partial(sim_fun, kwargs['alpha'])
        elif self.sim_func_choice == SimilarityFunction.tversky:
            sim_fun = partial(sim_fun, kwargs['alpha'], kwargs['beta'])

        sim_mtx = take_only_top_k(sim_fun(matrix.T), k=k)

        self.pred_mtx = matrix @ sim_mtx.T
        self.pred_mtx = self.pred_mtx.toarray()  # Not elegant but materializing the whole matrix makes the rest of the code easier to write/read

    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> typing.Union:
        out = self.pred_mtx[u_idxs[:, None], i_idxs]
        return out


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
