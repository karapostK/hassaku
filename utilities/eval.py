from collections import defaultdict

import bottleneck as bn
import numpy as np
import torch
from torch.utils.data import DataLoader

from algorithms.base_classes import RecommenderAlgorithm, SGDBasedRecommenderAlgorithm
from consts.consts import K_VALUES, SINGLE_SEED
from utilities.utils import reproducible, print_results


def Recall_at_k_batch(logits: np.ndarray, y_true: np.ndarray, k: int = 10, aggr_sum: bool = True):
    """
    Recall.
    :param logits: Logits. Shape is (batch_size, *).
    :param y_true: the true prediction. Shape is (batch_size, *)
    :param k: cut-off value
    :param aggr_sum: whether we sum the values over the batch_size. Default to true.

    :return: Recall@k. Shape is (batch_size,) if aggr_sum=False, otherwise returns a scalar.
    """

    n = logits.shape[0]
    dummy_column = np.arange(n).reshape(n, 1)

    idx_topk_part = bn.argpartition(-logits, k, axis=1)[:, :k]
    X_pred_binary = np.zeros_like(logits, dtype=bool)
    X_pred_binary[dummy_column, idx_topk_part] = True

    X_true_binary = y_true.astype(bool)

    num = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    den = X_true_binary.sum(axis=1)

    recall = np.zeros(n)
    # Avoiding division by 0
    indices = np.where(den != 0)
    recall[indices] = num[indices] / den[indices]

    assert (recall >= 0).all() and (recall <= 1).all()
    if aggr_sum:
        return np.sum(recall)
    else:
        return list(recall)


def Precision_at_k_batch(logits: np.ndarray, y_true: np.ndarray, k: int = 10, aggr_sum: bool = True):
    """
    Precision.
    :param logits: Logits. Shape is (batch_size, *).
    :param y_true: the true prediction. Shape is (batch_size, *)
    :param k: cut-off value
    :param aggr_sum: whether we sum the values over the batch_size. Default to true.

    :return: Precision@k. Shape is (batch_size,) if aggr_sum=False, otherwise returns a scalar.
    """

    n = logits.shape[0]
    dummy_column = np.arange(n).reshape(n, 1)

    idx_topk_part = bn.argpartition(-logits, k, axis=1)[:, :k]
    X_pred_binary = np.zeros_like(logits, dtype=bool)
    X_pred_binary[dummy_column, idx_topk_part] = True

    X_true_binary = y_true.astype(bool)
    num = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    precision = num / k

    assert (precision >= 0).all() and (precision <= 1).all()
    if aggr_sum:
        return np.sum(precision)
    else:
        return list(precision)


def NDCG_at_k_batch(logits: np.ndarray, y_true: np.ndarray, k: int = 10, aggr_sum: bool = True):
    """
    Normalized Discount Cumulative Gain. This implementation considers binary relevance.
    :param logits: Logits. Shape is (batch_size, *).
    :param y_true: the true prediction. Shape is (batch_size, *)
    :param k: cut-off value
    :param aggr_sum: whether we sum the values over the batch_size. Default to true.

    :return: NDCG@k. Shape is (batch_size,) if aggr_sum=False, otherwise returns a scalar.
    """

    n = logits.shape[0]
    dummy_column = np.arange(n).reshape(n, 1)

    idx_topk_part = bn.argpartition(-logits, k, axis=1)[:, :k]
    topk_part = logits[dummy_column, idx_topk_part]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk_ordered = idx_topk_part[dummy_column, idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (y_true[dummy_column, idx_topk_ordered] * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(int(y_true[i].sum()), k)]).sum() for i in range(n)])
    # the line above is just a fancy way to compute for each user the maximum IDCG by considering that all the top
    # positions (either all k, or the |relevant_items| positions) are taken by relevant items.

    NDCG = np.zeros(n)

    # Avoiding division by 0
    indices = np.where(IDCG != 0)
    NDCG[indices] = DCG[indices] / IDCG[indices]

    # there might be issues with the precision here, just enforcing a maximum to solve the problem
    indices = np.where(NDCG > 1)
    NDCG[indices] = 1.

    if aggr_sum:
        return np.sum(NDCG)
    else:
        return list(NDCG)


class FullEvaluator:
    """
    Helper class for the evaluation. When called with eval_batch, it updates the internal results. After the last batch,
    get_results will return the aggregated information for all users.
    """

    def __init__(self, n_users: int, aggr_users: bool = True):
        """

        :param aggr_users: Whether to aggregate the results for all users (with mean)
        """
        self.n_users = n_users
        self.aggr_users = aggr_users

        self.metrics_values = None

        self._reset_internal_dict()

    def _reset_internal_dict(self):
        self.metrics_values = defaultdict(lambda: 0) if self.aggr_users else defaultdict(list)

    def eval_batch(self, logits: np.ndarray, y_true: np.ndarray):
        """
        :param logits: Logits. Shape is (batch_size, *).
        :param y_true: the true prediction. Shape is (batch_size, *)
        """
        for k in K_VALUES:
            for metric_name, metric in zip(['precision@{}', 'recall@{}', 'ndcg@{}'],
                                           [Precision_at_k_batch, Recall_at_k_batch, NDCG_at_k_batch]):
                self.metrics_values[metric_name.format(k)] += metric(logits, y_true, k, self.aggr_users)

    def get_results(self):

        if self.aggr_users:
            for metric_name in self.metrics_values:
                self.metrics_values[metric_name] /= self.n_users

        metrics_dict = self.metrics_values
        self._reset_internal_dict()

        return metrics_dict


def evaluate_recommender_algorithm(alg: RecommenderAlgorithm, eval_loader: DataLoader, seed: int = SINGLE_SEED,
                                   device='cpu', rec_loss=None):
    reproducible(seed)

    evaluator = FullEvaluator(eval_loader.dataset.n_users)
    eval_loss = 0
    for u_idxs, i_idxs, labels in eval_loader:
        u_idxs = u_idxs.to(device)
        i_idxs = i_idxs.to(device).unsqueeze(-1)
        labels = labels.to(device)

        out = alg.predict(u_idxs, i_idxs).squeeze(-1)

        if isinstance(alg, SGDBasedRecommenderAlgorithm) and rec_loss is not None:
            eval_loss += rec_loss.compute_loss(out, labels).item()
            eval_loss += alg.get_and_reset_other_loss()

        if isinstance(out, torch.Tensor):
            out = out.to('cpu')

        evaluator.eval_batch(out, labels)

    eval_loss /= len(eval_loader)
    metrics_values = evaluator.get_results()
    if rec_loss is not None:
        metrics_values = {**metrics_values, 'eval_loss': eval_loss}

    print_results(metrics_values)
    return metrics_values
