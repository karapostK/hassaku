from enum import Enum

import numpy as np
import scipy as sc
from scipy import sparse as sp
from scipy.sparse import linalg as sp_linalg


def compute_jaccard_sim_mtx(matrix):
    jaccard_sim_mtx = (matrix @ matrix.T)

    counts = np.array(matrix.sum(axis=1)).squeeze()
    try:
        union = counts.T + counts - jaccard_sim_mtx  # may cause a memory error!
        jaccard_sim_mtx = sp.csr_matrix(jaccard_sim_mtx / union)
    except MemoryError as e:
        print('Resorting to slower method (never checked if it terminates though)')
        rows_nz, cols_nz = jaccard_sim_mtx.nonzero()
        jaccard_sim_mtx[rows_nz, cols_nz] = jaccard_sim_mtx[rows_nz, cols_nz] / (
                counts[rows_nz] + counts[cols_nz] - jaccard_sim_mtx[rows_nz, cols_nz])

    return jaccard_sim_mtx


def compute_cosine_sim_mtx(matrix):
    norms = sp_linalg.norm(matrix, axis=1)

    normalized_matrix = sp.csr_matrix((matrix.T / norms).T)

    cosine_sim_mtx = normalized_matrix @ normalized_matrix.T

    return cosine_sim_mtx


def compute_pearson_sim_mtx(matrix):
    means = np.array(matrix.mean(axis=1)).flatten()
    matrix_no_mean = matrix.copy().asfptype()

    for indx in range(matrix.shape[0]):
        matrix_no_mean.data[matrix.indptr[indx]:matrix.indptr[indx + 1]] -= means[indx]

    norms_no_mean = sp_linalg.norm(matrix_no_mean, axis=1)

    normalized_matrix_no_mean = sp.csr_matrix((matrix_no_mean.T / norms_no_mean).T)

    pearson_sim_mtx = normalized_matrix_no_mean @ normalized_matrix_no_mean.T

    return pearson_sim_mtx


def compute_pearson_dense_sim_mtx(matrix):
    means = np.array(matrix.mean(axis=1)).flatten()

    matrix_no_mean = matrix - means[:, None]

    norms_no_mean = sc.linalg.norm(matrix_no_mean, axis=1)

    normalized_matrix_no_mean = (matrix_no_mean.T / norms_no_mean).T

    pearson_dense_sim_mtx = sp.csr_matrix(normalized_matrix_no_mean @ normalized_matrix_no_mean.T)
    return pearson_dense_sim_mtx


def compute_asymmetric_cosine_sim_mtx(alpha, matrix):
    sums = np.squeeze(np.asarray(matrix.sum(axis=1)))

    sums_alpha = np.power(sums, alpha)
    sums_1_min_alpha = np.power(sums, 1 - alpha)

    denominator = np.outer(sums_alpha, sums_1_min_alpha)

    asymmetric_sim_mtx = sp.csr_matrix((matrix @ matrix.T) / denominator)
    return asymmetric_sim_mtx


def compute_sorensen_dice_sim_mtx(matrix):
    intersection = (matrix @ matrix.T)

    counts = matrix.sum(axis=1)
    counts_sum = counts + counts.T

    sorensedice_sim_mtx = sp.csr_matrix(2 * intersection / counts_sum)
    return sorensedice_sim_mtx


def compute_tversky_sim_mtx(alpha, beta, matrix):
    intersection = (matrix @ matrix.T)

    counts = matrix.sum(axis=1)
    complement = counts - intersection

    tversky_sim_mtx = sp.csr_matrix(intersection / (intersection + alpha * complement + beta * complement.T))
    return tversky_sim_mtx


class SimilarityFunction(Enum):
    #todo: the enum functionality is not used at all
    jaccard = compute_jaccard_sim_mtx
    cosine = compute_cosine_sim_mtx
    pearson = compute_pearson_sim_mtx
    asymmetric_cosine = compute_asymmetric_cosine_sim_mtx
    tversky = compute_tversky_sim_mtx
    sorensen_dice = compute_sorensen_dice_sim_mtx
