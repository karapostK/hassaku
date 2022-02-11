import bottleneck as bn
import numpy as np
import torch
from torch.utils.data import DataLoader

from algorithms.base_classes import RecommenderAlgorithm
from utilities.consts import K_VALUES, SINGLE_SEED
from utilities.enums import RecAlgorithmsEnum
from utilities.utils import reproducible, print_results


def Hit_Ratio_at_k_batch(logits: np.ndarray, k=10, sum=True):
    """
    Hit Ratio. It expects the positive logit in the first position of the vector.
    :param logits: Logits. Shape is (batch_size, n_neg + 1).
    :param k: threshold
    :param sum: if we have to sum the values over the batch_size. Default to true.
    :return: HR@K. Shape is (batch_size,) if sum=False, otherwise returns a scalar.
    """

    assert logits.shape[1] >= k, 'k value is too high!'

    idx_topk_part = bn.argpartition(-logits, k, axis=1)[:, :k]
    hrs = np.any(idx_topk_part[:] == 0, axis=1).astype(int)

    if sum:
        return np.sum(hrs)
    else:
        return hrs


def NDCG_at_k_batch(logits: np.ndarray, k=10, sum=True):
    """
    Normalized Discount Cumulative Gain. It expects the positive logit in the first position of the vector.
    :param logits: Logits. Shape is (batch_size, n_neg + 1).
    :param k: threshold
    :param sum: if we have to sum the values over the batch_size. Default to true.
    :return: NDCG@K. Shape is (batch_size,) if sum=False, otherwise returns a scalar.
    """
    assert logits.shape[1] >= k, 'k value is too high!'
    n = logits.shape[0]
    dummy_column = np.arange(n).reshape(n, 1)

    idx_topk_part = bn.argpartition(-logits, k, axis=1)[:, :k]
    topk_part = logits[dummy_column, idx_topk_part]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[dummy_column, idx_part]

    rows, cols = np.where(idx_topk == 0)
    ndcgs = np.zeros(n)

    if rows.size > 0:
        ndcgs[rows] = 1. / np.log2((cols + 1) + 1)

    if sum:
        return np.sum(ndcgs)
    else:
        return ndcgs


class Evaluator:
    """
    Helper class for the evaluation. When called with eval_batch, it updates the internal results. After the last batch,
    get_results will return the aggregated information for all users.
    """

    def __init__(self, n_users: int, logger=None):
        self.n_users = n_users
        self.logger = logger

        self.metrics_values = {}

    def eval_batch(self, out: np.ndarray, sum: bool = True):
        """
        :param out: Values after last layer. Shape is (batch_size, n_neg + 1).
        """
        for k in K_VALUES:
            for metric_name, metric in zip(['ndcg@{}', 'hit_ratio@{}'], [NDCG_at_k_batch, Hit_Ratio_at_k_batch]):
                if sum:
                    self.metrics_values[metric_name.format(k)] = self.metrics_values.get(metric_name.format(k), 0) + \
                                                                 metric(out, k)
                else:
                    self.metrics_values[metric_name.format(k)] = self.metrics_values.get(metric_name.format(k),
                                                                                         []) + list(
                        metric(out, k, False))

    def get_results(self, aggregated=True):
        """
        Returns the aggregated results (avg) and logs the results.
        """
        if aggregated:
            for metric_name in self.metrics_values:
                self.metrics_values[metric_name] /= self.n_users

            # Logging if logger is specified
            if self.logger:
                for metric_name in self.metrics_values:
                    self.logger.log_scalar(metric_name, self.metrics_values[metric_name])

        metrics_dict = self.metrics_values
        self.metrics_values = {}

        return metrics_dict


def evaluate_recommender_algorithm(alg: RecommenderAlgorithm, eval_loader: DataLoader, seed: int = SINGLE_SEED,
                                   device='cpu', rec_loss=None):
    reproducible(seed)

    evaluator = Evaluator(eval_loader.dataset.n_users)
    eval_loss = 0
    for u_idxs, i_idxs, labels in eval_loader:
        u_idxs = u_idxs.to(device)
        i_idxs = i_idxs.to(device)
        labels = labels.to(device)

        out = alg.predict(u_idxs, i_idxs)

        if rec_loss is not None:
            eval_loss += rec_loss.compute_loss(out, labels).item()
            eval_loss += alg.get_and_reset_other_loss()

        if isinstance(out, torch.Tensor):
            out = out.to('cpu')

        evaluator.eval_batch(out)

    eval_loss /= len(eval_loader)
    metrics_values = evaluator.get_results()
    if rec_loss is not None:
        metrics_values = {**metrics_values, 'eval_loss': eval_loss}

    print_results(metrics_values)
    return metrics_values


def evaluate_naive_algorithm(alg: RecommenderAlgorithm, eval_loader: DataLoader, seed: int = SINGLE_SEED):
    """
    Manual evaluation for Pop and Rand
    """
    reproducible(seed)

    metrics_values = {}

    for u_idxs, i_idxs, labels in eval_loader:

        for b in range(u_idxs.shape[0]):  # Going over the batch size

            if isinstance(alg, RecAlgorithmsEnum.rand.value):
                chosen_items = np.random.choice(np.arange(eval_loader.dataset.n_items), i_idxs.shape[1], replace=False)
            elif isinstance(alg, RecAlgorithmsEnum.pop.value):
                chosen_items = np.argsort(-eval_loader.dataset.pop_distribution)[:i_idxs.shape[1]]
            else:
                raise ValueError('Algorithm provided is non-compatible with this evaluation procedure!')

            for k in K_VALUES:
                position = np.where(chosen_items[:k] == i_idxs[b, 0].item())[0]
                u_hr = int(len(position) > 0)

                u_ndcg = 1 / (1 + np.log2(1 + position[0])) if len(position) > 0 else 0
                metrics_values[f'hit_ratio@{k}'] = metrics_values.get(f'hit_ratio@{k}', 0) + u_hr
                metrics_values[f'ndcg@{k}'] = metrics_values.get(f'ndcg@{k}', 0) + u_ndcg

    for metric_name in metrics_values:
        metrics_values[metric_name] /= eval_loader.dataset.n_users

    print_results(metrics_values)
    return metrics_values
