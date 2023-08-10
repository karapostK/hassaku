from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from algorithms.base_classes import SGDBasedRecommenderAlgorithm, RecommenderAlgorithm
from eval.eval_utils import K_VALUES
from eval.metrics import precision_at_k_batch, ndcg_at_k_batch, recall_at_k_batch
from utilities.utils import log_info_results


class FullEvaluator:
    """
    Helper class for the evaluation. It considers grouping at the level of the users. When called with eval_batch, it
    updates the internal results. After the last batch, get_results will return the metrics. It holds a special group
    with index -1 that is the "ALL" user group.
    """

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

    def eval_batch(self, u_idxs: torch.Tensor, logits: torch.Tensor, y_true: torch.Tensor):
        """
        :param u_idxs: User indexes. Shape is (batch_size).
        :param logits: Logits. Shape is (batch_size, n_items).
        :param y_true: the true prediction. Shape is (batch_size, n_items)
        """

        k_sorted_values = sorted(K_VALUES, reverse=True)
        k_max = k_sorted_values[0]
        idx_topk = logits.topk(k=k_max).indices
        self.n_entries[-1] += logits.shape[0]
        group_n_entries = defaultdict(int)

        for k in k_sorted_values:
            idx_topk = idx_topk[:, :k]
            for metric_name, metric in \
                    zip(['precision@{}', 'recall@{}', 'ndcg@{}'],
                        [precision_at_k_batch, recall_at_k_batch,
                         ndcg_at_k_batch]):
                metric_result = metric(logits=logits, y_true=y_true, k=k, aggr_sum=False,
                                       idx_topk=idx_topk).detach()  # Shape is (batch_size)

                # Collect results for 'all' users group
                self.group_metrics[-1][metric_name.format(k)] += \
                    metric_result.sum().item() if self.aggr_by_group else metric_result
                self.n_entries[-1] += logits.shape[0]

                # Collect results for specific user groups
                if self.n_groups > 0:
                    batch_user_to_user_groups = self.user_to_user_group[u_idxs]
                    for group_idx in range(self.n_groups):
                        group_metric_idx = np.where(batch_user_to_user_groups == group_idx)
                        group_metric = metric_result[group_metric_idx]
                        self.group_metrics[group_idx][metric_name.format(k)] += \
                            group_metric.sum().item() if self.aggr_by_group else group_metric
                        group_n_entries[group_idx] = group_metric.shape[0]

        if self.n_groups > 0:
            for group_idx in range(self.n_groups):
                self.n_entries[group_idx] += group_n_entries[group_idx]

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


def evaluate_recommender_algorithm(alg: RecommenderAlgorithm, eval_loader: DataLoader, device='cpu'):
    """
    Evaluation procedure that calls FullEvaluator on the dataset.
    """

    evaluator = FullEvaluator(aggr_by_group=True, n_groups=eval_loader.dataset.n_user_groups,
                              user_to_user_group=eval_loader.dataset.user_to_user_group)

    if not isinstance(alg, SGDBasedRecommenderAlgorithm):
        for u_idxs, i_idxs, labels in eval_loader:
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
        # We generate the item representation once (usually the bottleneck of evaluation)
        i_idxs = torch.arange(eval_loader.dataset.n_items).to(device)
        i_repr = alg.get_item_representations(i_idxs)

        for u_idxs, _, labels in eval_loader:
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
