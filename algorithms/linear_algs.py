import itertools
import logging
import multiprocessing
import os
import warnings

import numpy as np
from scipy import sparse as sp
from tqdm import trange

from algorithms.base_classes import SparseMatrixBasedRecommenderAlgorithm


class SLIM(SparseMatrixBasedRecommenderAlgorithm):

    def __init__(self, alpha: float, l1_ratio: float, max_iter: int):
        """
        SLIM: Sparse Linear Methods for Top-N Recommender Systems -  Xia Ning ; George Karypis (https://ieeexplore.ieee.org/document/6137254)
        Parallel implementation of SLIM algorithm (code from https://github.com/ruhan/toyslim/blob/master/slim_parallel.py)
        :param alpha: a + b (a and b are the multipliers of the L1 and L2 penalties, respectively)
        :param l1_ratio: a / (a + b) (a and b are the multipliers of the L1 and L2 penalties, respectively)
        :param max_iter: number of iterations to run max for each item
        """
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter

        self.pred_mtx = None

        self.name = 'SLIM'

        logging.info(f'Built {self.name} module \n'
                     f'- alpha: {self.alpha} \n'
                     f'- l1_ratio: {self.l1_ratio} \n'
                     f'- max_iter: {self.max_iter} \n')

    def fit(self, matrix: sp.spmatrix):

        warnings.simplefilter("ignore")

        matrix = sp.csc_matrix(matrix)

        n_items = matrix.shape[1]

        ranges = generate_slices(n_items)
        separated_tasks = []
        for from_j, to_j in ranges:
            separated_tasks.append([from_j, to_j, matrix.copy(), self.alpha, self.l1_ratio, self.max_iter])

        with multiprocessing.Pool() as pool:
            results = pool.map(SLIM.work, separated_tasks)

        W_rows_idxs = list(itertools.chain(*[x[0] for x in results]))
        W_cols_idxs = list(itertools.chain(*[x[1] for x in results]))
        W_data = list(itertools.chain(*[x[2] for x in results]))

        W = sp.csr_matrix((W_data, (W_rows_idxs, W_cols_idxs)), shape=(n_items, n_items))

        self.pred_mtx = matrix @ W
        self.pred_mtx = self.pred_mtx.toarray()

    @staticmethod
    def work(params):
        from sklearn.linear_model import ElasticNet

        from_j, to_j = params[0], params[1]
        A = params[2]
        alpha, l1_ratio, max_iter = params[3], params[4], params[5]

        elanet = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,  # Not considered by SLIM
            positive=True,  # Constraint in SLIM
            copy_X=False,  # efficiency reasons
            max_iter=max_iter,
            selection="random",  # efficiency reasons
            tol=1e-4  # assuming a good tolerance
        )

        W_rows_idxs = []
        W_cols_idxs = []
        W_data = []

        for j in trange(from_j, to_j, desc='{} -> {}'.format(from_j, to_j)):
            # Target column
            aj = A[:, j].toarray()

            # Removing the j-th item from all users
            # Need to zero the data entries related to the j-th column
            st_idx = A.indptr[j]
            en_idx = A.indptr[j + 1]

            copy = A.data[st_idx:en_idx].copy()
            A.data[st_idx:en_idx] = 0.0

            # Predicting the column
            elanet.fit(A, aj)

            # Fetching the coefficients (sparse)
            widx = elanet.sparse_coef_.indices
            wdata = elanet.sparse_coef_.data

            # Save information about position in the final matrix
            W_rows_idxs += list(widx)
            W_cols_idxs += [j] * len(widx)
            W_data += list(wdata)

            # reconstructing the matrix
            A.data[st_idx:en_idx] = copy

        return W_rows_idxs, W_cols_idxs, W_data

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
        return SLIM(conf['alpha'], conf['l1_ratio'], conf['max_iter'])


class EASE(SparseMatrixBasedRecommenderAlgorithm):

    def __init__(self, lam: int):
        """
        Embarassingly Shallow Autoencoders - Harald Steck (https://arxiv.org/abs/1905.03375)
        :param lam: lambda L2-norm regularization parameter
        """
        super().__init__()

        self.lam = lam

        self.pred_mtx = None

        self.name = 'EASE'

        logging.info(f'Built {self.name} module \n'
                     f'- lam: {self.lam} ')

    def fit(self, matrix: sp.spmatrix):
        # Computer Gram matrix
        G = matrix.transpose().dot(matrix).toarray()

        diagIndicies = np.diag_indices(G.shape[0])
        G[diagIndicies] += int(self.lam)

        P = np.linalg.inv(G)

        B = P / (-np.diag(P))
        B[diagIndicies] = 0

        self.pred_mtx = matrix @ B

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
        return EASE(conf['lam'])


def generate_slices(total_columns):
    """
    Generate slices that will be processed based on the number of cores
    available on the machine.
    """
    from multiprocessing import cpu_count

    cores = cpu_count()
    print('Running on {} cores'.format(cores))
    segment_length = total_columns // cores

    ranges = []
    now = 0

    while now < total_columns:
        end = now + segment_length

        # The last part can be a little greater that others in some cases, but
        # we can't generate more than #cores ranges
        end = end if end + segment_length <= total_columns else total_columns
        ranges.append((now, end))
        now = end

    return ranges
