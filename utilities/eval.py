from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from algorithms.base_classes import SGDBasedRecommenderAlgorithm, RecommenderAlgorithm
from consts.consts import K_VALUES, SINGLE_SEED
from utilities.utils import reproducible, print_results


def Recall_at_k_batch(logits: torch.Tensor, y_true: torch.Tensor, k: int = 10, aggr_sum: bool = True,
                      idx_topk: torch.Tensor = None):
    """
    Recall.
    :param logits: Logits. Shape is (batch_size, *).
    :param y_true: the true prediction. Shape is (batch_size, *)
    :param k: cut-off value
    :param aggr_sum: whether we sum the values over the batch_size. Default to true.
    :param idx_topk: pre-computed top-k indexes (used to index both logits and y_true)

    :return: Recall@k. Shape is (batch_size,) if aggr_sum=False, otherwise returns a scalar.
    """
    if idx_topk is not None:
        assert idx_topk is not None and idx_topk.shape[-1] == k, \
            'Top-k indexes have different "k" compared to the parameter function'
    idx_topk = logits.topk(k=k).indices if idx_topk is None else idx_topk
    indexing_column = torch.arange(logits.shape[0]).unsqueeze(-1)
    num = y_true[indexing_column, idx_topk].sum(dim=-1)
    den = y_true.sum(dim=-1)

    recall = num / den
    recall[torch.isnan(recall)] = .0

    assert (recall >= 0).all() and (recall <= 1).all()
    if aggr_sum:
        return recall.sum()
    else:
        return list(recall)


def Precision_at_k_batch(logits: torch.Tensor, y_true: torch.Tensor, k: int = 10, aggr_sum: bool = True,
                         idx_topk: torch.Tensor = None):
    """
    Precision.
    :param logits: Logits. Shape is (batch_size, *).
    :param y_true: the true prediction. Shape is (batch_size, *)
    :param k: cut-off value
    :param aggr_sum: whether we sum the values over the batch_size. Default to true.
    :param idx_topk: pre-computed top-k indexes (used to index both logits and y_true)

    :return: Precision@k. Shape is (batch_size,) if aggr_sum=False, otherwise returns a scalar.
    """

    if idx_topk is not None:
        assert idx_topk is not None and idx_topk.shape[-1] == k, \
            'Top-k indexes have different "k" compared to the parameter function'
    idx_topk = logits.topk(k=k).indices if idx_topk is None else idx_topk
    indexing_column = torch.arange(logits.shape[0]).unsqueeze(-1)
    num = y_true[indexing_column, idx_topk].sum(dim=-1)

    precision = num / k

    assert (precision >= 0).all() and (precision <= 1).all()
    if aggr_sum:
        return precision.sum()
    else:
        return list(precision)


def NDCG_at_k_batch(logits: torch.Tensor, y_true: torch.Tensor, k: int = 10, aggr_sum: bool = True,
                    idx_topk: torch.Tensor = None):
    """
    Normalized Discount Cumulative Gain. This implementation considers binary relevance.
    :param logits: Logits. Shape is (batch_size, *).
    :param y_true: the true prediction. Shape is (batch_size, *)
    :param k: cut-off value
    :param aggr_sum: whether we sum the values over the batch_size. Default to true.
    :param idx_topk: pre-computed top-k indexes (used to index both logits and y_true)

    :return: NDCG@k. Shape is (batch_size,) if aggr_sum=False, otherwise returns a scalar.
    """

    if idx_topk is not None:
        assert idx_topk is not None and idx_topk.shape[-1] == k, \
            'Top-k indexes have different "k" compared to the parameter function'
    idx_topk = logits.topk(k=k).indices if idx_topk is None else idx_topk
    indexing_column = torch.arange(logits.shape[0]).unsqueeze(-1)

    discount_template = 1. / torch.log2(torch.arange(2, k + 2).float())
    discount_template = discount_template.to(logits.device)

    DCG = (y_true[indexing_column, idx_topk] * discount_template).sum(-1)
    IDCG = (y_true.topk(k).values * discount_template).sum(-1)

    NDCG = DCG / IDCG
    NDCG[torch.isnan(NDCG)] = .0

    # there might be issues with the precision here, just enforcing a maximum to solve the problem
    NDCG = NDCG.clamp(max=1.)

    if aggr_sum:
        return NDCG.sum()
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

    def eval_batch(self, logits: torch.Tensor, y_true: torch.Tensor):
        """
        :param logits: Logits. Shape is (batch_size, *).
        :param y_true: the true prediction. Shape is (batch_size, *)
        """

        for k in K_VALUES:
            idx_topk = logits.topk(k=k).indices
            for metric_name, metric in zip(['precision@{}', 'recall@{}', 'ndcg@{}'],
                                           [Precision_at_k_batch, Recall_at_k_batch, NDCG_at_k_batch]):
                self.metrics_values[metric_name.format(k)] += metric(logits, y_true, k, self.aggr_users,
                                                                     idx_topk).detach()

    def get_results(self):

        for metric_name in self.metrics_values:
            if self.aggr_users:
                self.metrics_values[metric_name] = self.metrics_values[metric_name].item()
                self.metrics_values[metric_name] /= self.n_users
            else:
                self.metrics_values[metric_name] = self.metrics_values[metric_name].numpy()

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
        i_idxs = i_idxs.to(device)
        labels = labels.to(device)

        out = alg.predict(u_idxs, i_idxs)

        if not isinstance(out, torch.Tensor):
            out = torch.tensor(out).to(device)

        if isinstance(alg, SGDBasedRecommenderAlgorithm) and rec_loss is not None:
            eval_loss += rec_loss.compute_loss(out, labels).item()
            eval_loss += alg.get_and_reset_other_loss()

        evaluator.eval_batch(out, labels)

    eval_loss /= len(eval_loader)
    metrics_values = evaluator.get_results()
    if rec_loss is not None:
        metrics_values = {**metrics_values, 'eval_loss': eval_loss}

    print_results(metrics_values)
    return metrics_values
