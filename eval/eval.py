from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from algorithms.base_classes import SGDBasedRecommenderAlgorithm, RecommenderAlgorithm
from consts.consts import K_VALUES
from eval.metrics import precision_at_k_batch, ndcg_at_k_batch, recall_at_k_batch
from utilities.utils import print_results


class FullEvaluator:
    """
    Helper class for the evaluation. When called with eval_batch, it updates the internal results. After the last batch,
    get_results will return the aggregated information for all users.
    """

    def __init__(self, aggr_users: bool = True):
        """
        :param aggr_users: Whether to aggregate the results for all users (with mean)
        """
        self.aggr_users = aggr_users

        self.metrics_values = None
        self.n_entries = 0

        self._reset_internal_dict()

    def _reset_internal_dict(self):
        self.metrics_values = defaultdict(lambda: 0) if self.aggr_users else defaultdict(list)
        self.n_entries = 0

    def eval_batch(self, logits: torch.Tensor, y_true: torch.Tensor):
        """
        :param logits: Logits. Shape is (batch_size, *).
        :param y_true: the true prediction. Shape is (batch_size, *)
        """

        for k in K_VALUES:
            idx_topk = logits.topk(k=k).indices
            for metric_name, metric in \
                    zip(['precision@{}', 'recall@{}', 'ndcg@{}'],
                        [precision_at_k_batch, recall_at_k_batch, ndcg_at_k_batch]):
                self.metrics_values[metric_name.format(k)] += metric(logits, y_true, k, self.aggr_users,
                                                                     idx_topk).detach()
        self.n_entries += logits.shape[0]

    def get_results(self):

        for metric_name in self.metrics_values:
            if self.aggr_users:
                self.metrics_values[metric_name] = self.metrics_values[metric_name].item()
                self.metrics_values[metric_name] /= self.n_entries
            else:
                self.metrics_values[metric_name] = self.metrics_values[metric_name].numpy()

        metrics_dict = self.metrics_values
        self._reset_internal_dict()

        return metrics_dict


def evaluate_recommender_algorithm(alg: RecommenderAlgorithm, eval_loader: DataLoader, device='cpu'):
    """
    Evaluation procedure that calls FullEvaluator on the dataset.
    """

    evaluator = FullEvaluator()

    if not isinstance(alg, SGDBasedRecommenderAlgorithm):
        for u_idxs, i_idxs, labels in eval_loader:
            u_idxs = u_idxs.to(device)
            i_idxs = i_idxs.to(device)
            labels = labels.to(device)

            out = alg.predict(u_idxs, i_idxs)

            if not isinstance(out, torch.Tensor):
                out = torch.tensor(out).to(device)

            evaluator.eval_batch(out, labels)
    else:
        # We generate the item representation once (usually the bottleneck of evaluation)
        i_idxs = torch.arange(eval_loader.dataset.n_items).to(device)
        i_repr = alg.get_item_representations(i_idxs)

        for u_idxs, _, labels in eval_loader:
            u_idxs = u_idxs.to(device)
            labels = labels.to(device)

            u_repr = alg.get_user_representations(u_idxs)
            out = alg.combine_user_item_representations(u_repr, i_repr)

            evaluator.eval_batch(out, labels)

    metrics_values = evaluator.get_results()

    print_results(metrics_values)
    return metrics_values
