from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from algorithms.base_classes import SGDBasedRecommenderAlgorithm, RecommenderAlgorithm
from eval.metrics import precision_at_k_batch, ndcg_at_k_batch, recall_at_k_batch, hellinger_distance, \
    jensen_shannon_distance, kl_divergence
from utilities.utils import log_info_results


class FullEvaluator:
    """
    Helper class for the evaluation. It considers grouping at the level of the users. When called with eval_batch, it
    updates the internal results. After the last batch, get_results will return the metrics. It holds a special group
    with index -1 that is the "ALL" user group.
    """
    K_VALUES = [5, 10, 50, 100]  # K value for the evaluation metrics

    def __init__(self, aggr_by_group: bool = True, n_groups: int = 0, user_to_user_group: dict = None):
        """
        :param aggr_by_group: Whether to aggregate the results for all users within a group (with mean) or return the
        vectors
        :param n_groups: Number of user groups BEYOND 'ALL' USER GROUP. Defaults to 0. 'All' user group has index -1
        :param user_to_user_group: Mapping to user_idx to group_idx.
        """
        self.aggr_by_group = aggr_by_group
        self.n_groups = n_groups
        self.user_to_user_group = user_to_user_group

        self.group_metrics = None
        self.n_entries = None

        self._reset_internal_dict()

    def _reset_internal_dict(self):
        self.group_metrics = defaultdict(lambda: defaultdict(int) if self.aggr_by_group else defaultdict(list))
        self.n_entries = defaultdict(int)

    def _add_entry_to_dict(self, group_idx, metric_name, metric_result):
        if self.aggr_by_group:
            self.group_metrics[group_idx][metric_name] += metric_result.sum().item()
        else:
            self.group_metrics[group_idx][metric_name] += metric_result

    def get_n_groups(self):
        return self.n_groups

    def get_user_to_user_group(self):
        return self.user_to_user_group

    def eval_batch(self, u_idxs: torch.Tensor, logits: torch.Tensor, y_true: torch.Tensor):
        """
        :param u_idxs: User indexes. Shape is (batch_size).
        :param logits: Logits. Shape is (batch_size, n_items).
        :param y_true: the true prediction. Shape is (batch_size, n_items)
        """

        k_sorted_values = sorted(self.K_VALUES, reverse=True)
        k_max = k_sorted_values[0]
        idx_topk = logits.topk(k=k_max).indices

        # -- Counting per Group Entries --- #
        self.n_entries[-1] += logits.shape[0]
        if self.get_n_groups() > 0:
            batch_user_to_user_groups = self.get_user_to_user_group().to(u_idxs.device)[u_idxs]
            for group_idx in range(self.n_groups):
                group_metric_idx = torch.where(batch_user_to_user_groups == group_idx)[0]
                self.n_entries[group_idx] += len(group_metric_idx)

        # -- Computing the Metrics --- #
        for k in k_sorted_values:
            idx_topk = idx_topk[:, :k]

            for metric_name, metric in \
                    zip(
                        ['precision@{}', 'recall@{}', 'ndcg@{}'],
                        [precision_at_k_batch, recall_at_k_batch, ndcg_at_k_batch]
                    ):

                metric_result = metric(
                    logits=logits,
                    y_true=y_true,
                    k=k,
                    aggr_sum=False,
                    idx_topk=idx_topk
                ).detach()  # Shape is (batch_size)

                # Collect results for 'all' users group
                self._add_entry_to_dict(-1, metric_name.format(k), metric_result)

                # Collect results for specific user groups
                if self.get_n_groups() > 0:
                    batch_user_to_user_groups = self.get_user_to_user_group().to(u_idxs.device)[u_idxs]
                    for group_idx in range(self.n_groups):
                        group_metric_idx = np.where(batch_user_to_user_groups.cpu() == group_idx)
                        self._add_entry_to_dict(group_idx, metric_name.format(k), metric_result[group_metric_idx])

    def get_results(self):
        metrics_dict = dict()
        for group_idx in self.group_metrics:
            for metric_name in self.group_metrics[group_idx]:
                if group_idx != -1:
                    final_metric_name = 'group_' + str(group_idx) + '_' + metric_name
                else:
                    final_metric_name = metric_name
                if self.aggr_by_group:
                    metrics_dict[final_metric_name] = self.group_metrics[group_idx][metric_name] / self.n_entries[
                        group_idx]
                else:
                    metrics_dict[final_metric_name] = torch.stack(
                        self.group_metrics[group_idx][metric_name]).cpu().numpy()

        self._reset_internal_dict()

        return metrics_dict


class FullEvaluatorCalibrationDecorator(FullEvaluator):
    """
    Class that decorates a FullEvaluator object to add calibration metrics.
    """

    CALIBRATION_K_VALUES = [5, 10, 50, 100]

    def __init__(self, full_evaluator: FullEvaluator, item_tag_mtx: torch.Tensor, user_tag_mtx: torch.Tensor,
                 metric_name_prefix: str = 'tag',
                 beta_smoothening: float = .01):

        """
        @param full_evaluator: FullEvaluator object to decorate
        @param item_tag_mtx: Tensor of shape [n_items,n_tags]. The i-th row is the normalized distribution of the
        i-th item over the n_tags.
        @param user_tag_mtx: Tensor of shape [n_users,n_tags]. The i-th row is the frequency distribution of the
        i-th user over the n_tags.
        @param beta_smoothening: Smoothening value applied to the recommendation. See Eq. 5 in H. Steck Calibrated Recommendations

        """
        assert 0 <= beta_smoothening <= 1, 'Beta value out of bounds'

        self.full_evaluator = full_evaluator
        self.item_tag_mtx = item_tag_mtx
        self.user_tag_mtx = user_tag_mtx
        self.metric_name_prefix = metric_name_prefix
        self.beta_smoothening = beta_smoothening

    def _reset_internal_dict(self):
        self.full_evaluator._reset_internal_dict()

    def _add_entry_to_dict(self, group_idx, metric_name, metric_result):
        self.full_evaluator._add_entry_to_dict(group_idx, metric_name, metric_result)

    def get_n_groups(self):
        return self.full_evaluator.get_n_groups()

    def get_user_to_user_group(self):
        return self.full_evaluator.get_user_to_user_group()

    def get_results(self):
        return self.full_evaluator.get_results()

    def eval_batch(self, u_idxs: torch.Tensor, logits: torch.Tensor, y_true: torch.Tensor):
        self.full_evaluator.eval_batch(u_idxs, logits, y_true)

        self.user_tag_mtx = self.user_tag_mtx.to(logits.device)
        self.item_tag_mtx = self.item_tag_mtx.to(logits.device)

        k_sorted_values = sorted(FullEvaluatorCalibrationDecorator.CALIBRATION_K_VALUES, reverse=True)
        k_max = k_sorted_values[0]
        idx_topk = logits.topk(k=k_max).indices

        batch_user_train_tags_frequency = self.user_tag_mtx[u_idxs]

        for k in k_sorted_values:
            idx_topk = idx_topk[:, :k]

            # Compute recommendation distribution
            batch_user_top_items_tags = self.item_tag_mtx[idx_topk]  # [batch_size, top_k, n_tags]
            batch_user_tags_frequency = batch_user_top_items_tags.sum(1)  # [batch_size, n_tags]
            batch_user_tags_frequency /= k

            # Smoothening the batch frequency (Eq.5 in Calibrated Recommendation by H. Steck)
            batch_user_tags_frequency = self.beta_smoothening * batch_user_train_tags_frequency + (
                    1 - self.beta_smoothening) * batch_user_tags_frequency

            for metric_name, metric in \
                    zip(
                        ['hellinger_distance@{}', 'jensen_shannon_distance@{}', 'kl_divergence@{}'],
                        [hellinger_distance, jensen_shannon_distance, kl_divergence]
                    ):

                metric_name = self.metric_name_prefix + '_' + metric_name.format(k)
                # Shape is [batch_size].
                # N.B. kl divergence requires the target distribution as first argument!
                metric_result = metric(batch_user_train_tags_frequency, batch_user_tags_frequency).detach()

                # Collect results for 'all' users group
                self._add_entry_to_dict(-1, metric_name, metric_result)

                # Collect results for specific user groups
                if self.get_n_groups() > 0:
                    batch_user_to_user_groups = self.get_user_to_user_group()[u_idxs]
                    for group_idx in range(self.get_n_groups()):
                        group_metric_idx = np.where(batch_user_to_user_groups == group_idx)
                        group_metric = metric_result[group_metric_idx]
                        self._add_entry_to_dict(group_idx, metric_name, group_metric)


def evaluate_recommender_algorithm(alg: RecommenderAlgorithm, eval_loader: DataLoader, evaluator: FullEvaluator,
                                   device='cpu', verbose=False):
    """
    Evaluation procedure that calls FullEvaluator on the dataset.
    """

    if verbose:
        iterator = tqdm(eval_loader)
    else:
        iterator = eval_loader

    if not isinstance(alg, SGDBasedRecommenderAlgorithm):
        for u_idxs, i_idxs, labels in iterator:
            u_idxs = u_idxs.to(device)
            i_idxs = i_idxs.to(device)
            labels = labels.to(device)

            out = alg.predict(u_idxs, i_idxs)

            batch_mask = torch.tensor(eval_loader.dataset.exclude_data[u_idxs.cpu()].A)
            out[batch_mask] = -torch.inf

            if not isinstance(out, torch.Tensor):
                out = torch.tensor(out).to(device)

            evaluator.eval_batch(u_idxs, out, labels)
    else:
        with torch.no_grad():
            # We generate the item representation once (usually the bottleneck of evaluation)
            i_idxs = torch.arange(eval_loader.dataset.n_items).to(device)
            i_repr = alg.get_item_representations(i_idxs)

            for u_idxs, _, labels in iterator:
                u_idxs = u_idxs.to(device)
                labels = labels.to(device)

                u_repr = alg.get_user_representations(u_idxs)
                out = alg.combine_user_item_representations(u_repr, i_repr)

                batch_mask = torch.tensor(eval_loader.dataset.exclude_data[u_idxs.cpu()].A)
                out[batch_mask] = -torch.inf

                evaluator.eval_batch(u_idxs, out, labels)

    metrics_values = evaluator.get_results()

    log_info_results(metrics_values)
    return metrics_values
