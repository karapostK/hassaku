from enum import Enum

import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg as sp_linalg

from utilities.utils import FunctionWrapper

"""
NB. The following similarities functions have been considered for implicit data! All of them assume that matrix is 
a sparse matrix with 0s and 1s. 
NB. The following code assumes that:
1) The whole sparse matrix, after computing the similarities, can be stored in memory.
2) All the non-zero entries before normalization (so the numerator of the similarities) can be stored in memory
The code could be potentially be optimized by considering iterative block computation, merging the similarity computing
with the "take_only_top_k" from knn_algs.py. However, for the hardware at my disposal this is not needed.
"""


def compute_jaccard_sim_mtx(matrix):
    counts = np.array(matrix.sum(axis=1)).squeeze()

    jaccard_sim_mtx = sp.coo_matrix(matrix @ matrix.T)

    jaccard_sim_mtx.data = jaccard_sim_mtx.data / (
            counts[jaccard_sim_mtx.row] + counts[jaccard_sim_mtx.col] - jaccard_sim_mtx.data)

    jaccard_sim_mtx = sp.csr_matrix(jaccard_sim_mtx)

    return jaccard_sim_mtx


def compute_cosine_sim_mtx(matrix):
    norms = sp_linalg.norm(matrix, axis=1)

    cosine_sim_mtx = sp.coo_matrix(matrix @ matrix.T)

    cosine_sim_mtx.data = cosine_sim_mtx.data / (norms[cosine_sim_mtx.row] * norms[cosine_sim_mtx.col])
    cosine_sim_mtx = sp.csr_matrix(cosine_sim_mtx)

    return cosine_sim_mtx


def compute_pearson_sim_mtx(matrix):
    means = np.array(matrix.mean(axis=1)).flatten()
    pearson_sim_mtx = matrix.copy().asfptype()

    for indx in range(matrix.shape[0]):
        pearson_sim_mtx.data[matrix.indptr[indx]:matrix.indptr[indx + 1]] -= means[indx]

    norms_no_mean = sp_linalg.norm(pearson_sim_mtx, axis=1)

    pearson_sim_mtx = sp.coo_matrix(pearson_sim_mtx @ pearson_sim_mtx.T)
    pearson_sim_mtx.data = pearson_sim_mtx.data / (
            norms_no_mean[pearson_sim_mtx.row] * norms_no_mean[pearson_sim_mtx.col])

    pearson_sim_mtx = sp.csr_matrix(pearson_sim_mtx)
    return pearson_sim_mtx


def compute_asymmetric_cosine_sim_mtx(alpha, matrix):
    sums = np.squeeze(np.asarray(matrix.sum(axis=1)))

    sums_alpha = np.power(sums, alpha)
    sums_1_min_alpha = np.power(sums, 1 - alpha)

    asymmetric_sim_mtx = sp.coo_matrix(matrix @ matrix.T)
    asymmetric_sim_mtx.data = asymmetric_sim_mtx.data / (
            sums_alpha[asymmetric_sim_mtx.row] * sums_1_min_alpha[asymmetric_sim_mtx.col])

    asymmetric_sim_mtx = sp.csr_matrix(asymmetric_sim_mtx)
    return asymmetric_sim_mtx


def compute_sorensen_dice_sim_mtx(matrix):
    counts = np.array(matrix.sum(axis=1)).squeeze()

    sorensedice_sim_mtx = sp.coo_matrix(matrix @ matrix.T)

    sorensedice_sim_mtx.data = 2 * sorensedice_sim_mtx.data / (
            counts[sorensedice_sim_mtx.row] + counts[sorensedice_sim_mtx.col])

    sorensedice_sim_mtx = sp.csr_matrix(sorensedice_sim_mtx)
    return sorensedice_sim_mtx


def compute_tversky_sim_mtx(alpha, beta, matrix):
    counts = np.array(matrix.sum(axis=1)).squeeze()

    tversky_sim_mtx = sp.coo_matrix(matrix @ matrix.T)

    tversky_sim_mtx.data = tversky_sim_mtx.data / (
            tversky_sim_mtx.data + alpha * (counts[tversky_sim_mtx.row] - tversky_sim_mtx.data) + beta * (
            counts[tversky_sim_mtx.col] - tversky_sim_mtx.data))

    tversky_sim_mtx = sp.csr_matrix(tversky_sim_mtx)
    return tversky_sim_mtx


class SimilarityFunctionEnum(Enum):
    jaccard = FunctionWrapper(compute_jaccard_sim_mtx)
    cosine = FunctionWrapper(compute_cosine_sim_mtx)
    pearson = FunctionWrapper(compute_pearson_sim_mtx)
    asymmetric_cosine = FunctionWrapper(compute_asymmetric_cosine_sim_mtx)
    tversky = FunctionWrapper(compute_tversky_sim_mtx)
    sorensen_dice = FunctionWrapper(compute_sorensen_dice_sim_mtx)
