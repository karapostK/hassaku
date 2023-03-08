import math
from enum import Enum

import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg as sp_linalg
from tqdm import trange

from utilities.utils import FunctionWrapper

"""
NB. The following similarities functions have been considered for implicit data! All of them assume that matrix is 
a sparse matrix with 0s and 1s. 
NB. The following code assumes that the whole sparse matrix, after computing the similarities, can be stored in memory.
"""


def compute_similarity_top_k(matrix, sim_function, k, block_size=6048):
    """
    Computes the similarity matrix from the given matrix (each row is considered as an entity).
    It keeps only the k highest similarities.

    """

    n_entities = matrix.shape[0]
    steps = math.ceil(n_entities / block_size)

    new_data = []
    new_indices = []
    new_indptr = [0]

    cumulative_sum = 0

    for step in trange(steps):
        sub_mtx = matrix[step * block_size: (step + 1) * block_size, :]
        sub_sim_mtx = sim_function(matrix, sub_mtx, step, block_size)

        for idx in range(sub_sim_mtx.shape[0]):
            start_idx = sub_sim_mtx.indptr[idx]
            end_idx = sub_sim_mtx.indptr[idx + 1]

            data = sub_sim_mtx.data[start_idx:end_idx]
            ind = sub_sim_mtx.indices[start_idx:end_idx]

            # Avoiding taking the user/item itself
            if len(data) > 0:
                # The if is there to avoid cases where there are no closest neighbours (so even the sim to itself is 0)
                self_idx = np.where(ind == (idx + step * block_size))[0][0]
                data[self_idx] = 0.

            top_k_indxs = np.argsort(-data)[:k]

            top_k_data = data[top_k_indxs]
            top_k_indices = ind[top_k_indxs]

            new_data += list(top_k_data)
            new_indices += list(top_k_indices)
            cumulative_sum += len(top_k_data)
            new_indptr.append(cumulative_sum)

    return sp.csr_matrix((new_data, new_indices, new_indptr), shape=(n_entities, n_entities))


def compute_jaccard_sim_mtx(matrix, sub_mtx, step, block_size):
    counts = np.array(matrix.sum(axis=1)).squeeze()

    sub_mtx = sp.coo_matrix(sub_mtx @ matrix.T)
    sub_mtx.data /= (counts[sub_mtx.row + (step * block_size)] + counts[
        sub_mtx.col] - sub_mtx.data)

    return sp.csr_matrix(sub_mtx)


def compute_cosine_sim_mtx(matrix, sub_mtx, step, block_size):
    norms = sp_linalg.norm(matrix, axis=1)

    sub_mtx = sp.coo_matrix(sub_mtx @ matrix.T)
    sub_mtx.data = sub_mtx.data / (norms[sub_mtx.row + (step * block_size)] * norms[sub_mtx.col])

    return sp.csr_matrix(sub_mtx)


def compute_asymmetric_cosine_sim_mtx(alpha, matrix, sub_mtx, step, block_size):
    sums = np.squeeze(np.asarray(matrix.sum(axis=1)))

    sums_alpha = np.power(sums, alpha)
    sums_1_min_alpha = np.power(sums, 1 - alpha)

    sub_mtx = sp.coo_matrix(sub_mtx @ matrix.T)
    sub_mtx.data /= (
            sums_alpha[sub_mtx.row + (step * block_size)] * sums_1_min_alpha[sub_mtx.col])

    return sp.csr_matrix(sub_mtx)


def compute_sorensen_dice_sim_mtx(matrix, sub_mtx, step, block_size):
    counts = np.array(matrix.sum(axis=1)).squeeze()

    sub_mtx = sp.coo_matrix(sub_mtx @ matrix.T)

    sub_mtx.data /= (counts[sub_mtx.row + (step * block_size)] + counts[sub_mtx.col])
    sub_mtx.data *= 2

    return sp.csr_matrix(sub_mtx)


def compute_tversky_sim_mtx(alpha, beta, matrix, sub_mtx, step, block_size):
    counts = np.array(matrix.sum(axis=1)).squeeze()

    sub_mtx = sp.coo_matrix(sub_mtx @ matrix.T)

    sub_mtx.data /= (sub_mtx.data + alpha * (counts[sub_mtx.row + (step * block_size)] - sub_mtx.data) + beta * (
            counts[sub_mtx.col] - sub_mtx.data))

    return sp.csr_matrix(sub_mtx)


class SimilarityFunctionEnum(Enum):
    jaccard = FunctionWrapper(compute_jaccard_sim_mtx)
    cosine = FunctionWrapper(compute_cosine_sim_mtx)
    asymmetric_cosine = FunctionWrapper(compute_asymmetric_cosine_sim_mtx)
    tversky = FunctionWrapper(compute_tversky_sim_mtx)
    sorensen_dice = FunctionWrapper(compute_sorensen_dice_sim_mtx)
