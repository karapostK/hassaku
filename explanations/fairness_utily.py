import glob
import json
import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
import wandb
from paramiko import SSHClient
from scp import SCPClient
from torch import Tensor, nn
from torch.utils import data

from conf.conf_parser import parse_conf_file
from eval.eval import FullEvaluator
from eval.metrics import weight_ndcg_at_k_batch
from train.rec_losses import RecommenderSystemLoss
from wandb_conf import ENTITY_NAME, PROJECT_NAME

ALPHAS = [.9, .7, .5, .3, .1, .0]
TOP_K_RECOMMENDATIONS = 10


def compute_rec_gap(results_dict: dict, metric_name: str = 'ndcg@10', return_perc: bool = True):
    """
    Easy function to compute the percentage difference, respect to the mean, of the difference between group_0 and group_1
    NB. The value returned has a sign and might also return a negative percentage. This represents the case where group_1
    receives better values than group 0.
    @param results_dict:
    @param metric_name:
    @param return_perc:
    @return:
    """
    rec_gap = (results_dict['group_0_' + metric_name] - results_dict['group_1_' + metric_name]) / results_dict[
        metric_name]
    if return_perc:
        return 100 * rec_gap
    else:
        return rec_gap


def compute_tag_user_frequencies(users_top_items, item_idxs_to_tags, user_idxs_to_group,
                                 reduce_weight_multiple_tags=True):
    """
    The function returns user-group tags frequencies of their recommendations. This is computed as follows:
    - Each user's recommendations are mapped to tags. If an item has multiple tags (and reduce_weight_multiple_tags=True)
        then each tags gets 1/len(tags) for weight.
    - The user's tag frequencies are computed accordingly by simply aggregating the tags from the recommendations.
    - The frequencies are averaged on group level.

    @param users_top_items: [n_users, *] Represents the recommendations for each user of the system. Each entry is an item id
    @param item_idxs_to_tags: Maps the item_idx to a list of tags
    @param user_idxs_to_group: Maps the user_idx to groups
    @param reduce_weight_multiple_tags: Whether to consider 1/len(tags) as weight when an item has multiple tags
    @return: group frequencies.
    """

    # Getting all the tags
    tags_names = sorted(list(set(item_idxs_to_tags.explode())))
    n_groups = user_idxs_to_group.max() + 1

    # Cumulative frequencies
    groups_cumulative_tag_frequencies = []
    for group_idxs in range(n_groups):
        groups_cumulative_tag_frequencies.append(pd.Series(data=[0.] * len(tags_names), index=tags_names,
                                                           name='tags'))

    for user_idx, user_top_items in users_top_items:
        # Tags of the top items
        n_items = len(user_top_items)
        user_top_items_tags = item_idxs_to_tags.loc[user_top_items]

        if reduce_weight_multiple_tags:
            # Giving smaller weights to the tags of an items when the items itself has multiple tags.
            # e.g. i_1 with genre 'Drama', 'Drama' gets weight of 1.
            # e.g. i_2 with genres 'Drama, Action' then both genres weight 0.5 instead of 1.
            # This partially counteract the interactions of tags appearing in every movie
            user_top_items_tags_lens = user_top_items_tags.apply(len).array
            user_top_items_tags_lens = user_top_items_tags_lens.repeat(user_top_items_tags_lens)
            tags_normalizing_values = user_top_items_tags_lens
        else:
            tags_normalizing_values = 1

        user_top_items_tags = user_top_items_tags.explode().to_frame()
        user_top_items_tags['frequency'] = 1 / tags_normalizing_values

        user_top_items_tags_frequencies = user_top_items_tags.groupby('tags').aggregate(sum).frequency

        if reduce_weight_multiple_tags:
            user_top_items_tags_frequencies /= n_items
        else:
            user_top_items_tags_frequencies /= len(user_top_items_tags_frequencies)

        user_group_idx = user_idxs_to_group.loc[user_idx]

        groups_cumulative_tag_frequencies[user_group_idx] = groups_cumulative_tag_frequencies[user_group_idx].add(
            user_top_items_tags_frequencies, fill_value=0)

    # Normalize by # of users
    for group_idxs in range(n_groups):
        groups_cumulative_tag_frequencies[group_idxs] = groups_cumulative_tag_frequencies[group_idxs] / \
                                                        user_idxs_to_group.value_counts().loc[group_idxs]

    return groups_cumulative_tag_frequencies


def multiply_mask(in_repr, multiplication_mask, in_idxs=None, only_idxs=None):
    """
    Multiplies the multiplication mask on the input representations in_repr. If nothing else is specified, then the mask
    is applied to each entry in in_repr. If both in_idxs and only_idxs are provided then the masking might be applied to
    the entries of in_idxs IF they are in only_idxs.
    @param in_repr: Users/Item representations. Shape is [n_entries, n_prototypes]
    @param multiplication_mask: Multiplication mask. Default values should be 1. Shape is [n_prototypes]
    @param in_idxs:  Indexes of the Users/Items representation appearing in in_repr. Shape is [n_entries]
    @param only_idxs: Which entries should be considered for the multiplication. Shape is
    @return:
    """

    if only_idxs is None:
        entry_mask = torch.ones(in_repr.shape[0], dtype=bool).to(in_repr.device)
    else:
        if in_idxs is None:
            raise ValueError('in_idxs cannot be None if only_idxs is not None')
        else:
            entry_mask = torch.isin(in_idxs, only_idxs, assume_unique=True)

    new_in_repr = in_repr.clone()
    new_in_repr[entry_mask] *= multiplication_mask
    # Make sure that the output is still consistent with the model!
    return new_in_repr


def hellinger_distance(p, q):
    summed = (torch.square(torch.sqrt(p) - torch.sqrt(q))).sum(-1)
    return 0.5 * torch.sqrt(summed)


def jensen_shannon_distance(p, q):
    # https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    m_log = (0.5 * (p + q)).log()

    first_term = (p * (p.log() - m_log)).sum(-1)
    second_term = (q * (q.log() - m_log)).sum(-1)
    jsd = 0.5 * (first_term + second_term)
    return torch.sqrt(jsd)


def kl_divergence(true_p, model_p):
    # https://dl.acm.org/doi/pdf/10.1145/3240323.3240372
    return (true_p * (true_p.log() - model_p.log())).sum(-1)


class NeuralSort(torch.nn.Module):
    """
    Code is from https://github.com/ermongroup/neuralsort/blob/master/pytorch/neuralsort.py.
    Paper is STOCHASTIC OPTIMIZATION OF SORTING NETWORKS VIA CONTINUOUS RELAXATIONS by Grover et al. ICLR 2019
    """

    def __init__(self, tau=1.0, hard=False):
        super(NeuralSort, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n x 1
        """
        scores = scores.unsqueeze(-1)
        bsize = scores.size()[0]
        dim = scores.size()[1]
        one = torch.cuda.FloatTensor(dim, 1).fill_(1)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(
            one, torch.transpose(one, 0, 1)))
        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)
                   ).type(torch.cuda.FloatTensor)
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C - B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)

        if self.hard:
            P = torch.zeros_like(P_hat, device='cuda')
            b_idx = torch.arange(bsize).repeat([1, dim]).view(dim, bsize).transpose(
                dim0=1, dim1=0).flatten().type(torch.cuda.LongTensor)
            r_idx = torch.arange(dim).repeat(
                [bsize, 1]).flatten().type(torch.cuda.LongTensor)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat


class ConcatDataLoaders:
    def __init__(self, dataloader_0: torch.utils.data.DataLoader, dataloader_1: torch.utils.data.DataLoader):
        self.dataloader_0 = dataloader_0
        self.dataloader_1 = dataloader_1
        self.zipped_dataloader = None

    def __iter__(self):
        self.zipped_dataloader = zip(self.dataloader_0, self.dataloader_1)
        return self

    def __next__(self):
        (u_idxs_0, i_idxs_0, labels_0), (u_idxs_1, i_idxs_1, labels_1) = self.zipped_dataloader.__next__()
        u_idxs = torch.cat([u_idxs_0, u_idxs_1], dim=0)
        i_idxs = torch.cat([i_idxs_0, i_idxs_1], dim=0)
        labels = torch.cat([labels_0, labels_1], dim=0)
        return u_idxs, i_idxs, labels

    def __len__(self):
        return min(len(self.dataloader_0), len(self.dataloader_1))


class RecGapLoss(RecommenderSystemLoss):
    """
    N.B. This class assumes only 2 groups.
    N.B. This class is based on NDCG. However, other ranking functions could be used.
    """

    def __init__(self, n_items: int = None, aggregator: str = 'mean', train_neg_strategy: str = 'uniform',
                 neg_train: int = 4, fairness_at_k: int = 10, start_group_1_in_batch: int = None,
                 neural_sort_tau: float = 1.):
        """
        @param fairness_at_k: k at which the recgap is measured
        @param start_group_1_in_batch: index, within the batch, where the first person starts.
        @param neural_sort_tau: tau to be used in the NeuralSort layer.
        """
        assert start_group_1_in_batch is not None
        super().__init__(n_items, aggregator, train_neg_strategy, neg_train)
        self.fairness_at_k = fairness_at_k
        self.start_group_1_in_batch = start_group_1_in_batch  # Used to split the batch in two.
        self.neural_sort_tau = neural_sort_tau

        self.neural_sort_layer = NeuralSort(self.neural_sort_tau)
        self.mse_loss = nn.MSELoss()

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        return RecGapLoss(fairness_at_k=conf['fairness_at_k'],
                          start_group_1_in_batch=conf['start_group_1_in_batch'],
                          neural_sort_tau=conf['neural_sort_tau'],
                          )

    def compute_loss(self, scores, labels):
        assert scores.shape[-1] >= self.fairness_at_k, \
            f"There are not too many items in the batch. It is not possible to measure recgap at k={self.fairness_at_k}"

        # Compute Metric-dependent weight
        weight_ndcg = weight_ndcg_at_k_batch(labels, k=self.fairness_at_k)

        # Compute the Hit Function
        # The formula of approx hit works in the following way:
        # 1) All 'true' items are in the first positions of the scores vector.
        # 2) The [:,None,:] broadcasting allows us to apply the label mask to each row of the sorting matrix
        # This is also equivalent on zeroing out the right side of the matrix. We only care about the positions of the
        # true items anyway
        # 3) Multiplication is carried out (again we care only about the true items)
        # 4) We sum over the last column, which is equivalent to see if we 'hit' the item at that position.
        approx_sorting_matrix = self.neural_sort_layer(scores)  # [batch_size, n_scores (positions), n_scores (index) ]
        apporx_hit = (approx_sorting_matrix * labels[:, None, :]).sum(-1)  # [batch_size, n_scores (positions)]

        # Compute the NDCG by multiplying the weights
        apporx_hit = apporx_hit[:, :self.fairness_at_k]
        approx_ndcg = (apporx_hit * weight_ndcg).sum(-1)

        # Compute RecGap as difference of two means
        # || group_0_ndcg - group_1_ndcg||^2

        group_0_approx_ndcg = approx_ndcg[:self.start_group_1_in_batch]
        group_1_approx_ndcg = approx_ndcg[self.start_group_1_in_batch:]

        avg_group_0_ndcg = group_0_approx_ndcg.mean()
        avg_group_1_ndcg = group_1_approx_ndcg.mean()

        rec_gap_loss = self.mse_loss(avg_group_0_ndcg, avg_group_1_ndcg)

        return rec_gap_loss


def fetch_best_in_sweep(sweep_id, good_faith=True, preamble_path=None, project_base_directory: str = '..',
                        wandb_entitiy_name=ENTITY_NAME, wandb_project_name=PROJECT_NAME):
    """
    It returns the configuration of the best model of a specific sweep.
    However, since private wandb projects can only be accessed by 1 member, sharing of the models is basically impossible.
    The alternative, with good_faith=True it simply looks at the local directory with that specific sweep and hopes there is a model there.
    If there are multiple models then it raises an error.

    @param sweep_id:
    @param good_faith:  whether to look only at local folders or not
    @param preamble_path: if specified it will replace the part that precedes hassaku (e.g. /home/giovanni/hassaku... -> /something_else/hassaku/....
    @param project_base_directory: where is the project directory (either relative from where the code is running or in absolute path.
    @return:
    """
    if good_faith:
        sweep_path = glob.glob(f'{project_base_directory}/saved_models/*/sweeps/{sweep_id}')
        if len(sweep_path) > 1:
            raise ValueError('There should not be two sweeps with the same id')
        sweep_path = sweep_path[0]
        best_run_path = os.listdir(sweep_path)
        if len(best_run_path) > 1:
            raise ValueError('There are more than 1 runs in the project, which one is the best?')

        best_run_path = best_run_path[0]
        best_run_config = parse_conf_file(os.path.join(sweep_path, best_run_path, 'conf.yml'))

    else:
        api = wandb.Api()
        sweep = api.sweep(f"{wandb_entitiy_name}/{wandb_project_name}/{sweep_id}")

        best_run = sweep.best_run()
        best_run_host = best_run.metadata['host']
        best_run_config = json.loads(best_run.json_config)
        if '_items' in best_run_config:
            best_run_config = best_run_config['_items']['value']
        else:
            best_run_config = {k: v['value'] for k, v in best_run_config.items()}

        best_run_model_path = best_run_config['model_path']
        print('Best Run Model Path: ', best_run_model_path)

        # Create base directory if absent
        local_path = os.path.join(project_base_directory, best_run_model_path)
        current_host = socket.gethostname()

        if not os.path.isdir(local_path):
            Path(local_path).mkdir(parents=True, exist_ok=True)

            if current_host != best_run_host:
                print(f'Importing Model from {best_run_host}')
                # Moving the best model to local directory
                # N.B. Assuming same username
                with SSHClient() as ssh:
                    ssh.load_system_host_keys()
                    ssh.connect(best_run_host)

                    with SCPClient(ssh.get_transport()) as scp:
                        # enoughcool4hardcoding
                        dir_path = "hassaku"
                        if best_run_host == 'passionpit.cp.jku.at':
                            dir_path = os.path.join(dir_path, "PycharmProjects")

                        scp.get(remote_path=os.path.join(dir_path, best_run_model_path),
                                local_path=os.path.dirname(local_path),
                                recursive=True)
            else:
                raise FileNotFoundError(f"The model should be local but it was not found! Path is: {local_path}")

    if preamble_path:
        pre, post = best_run_config['dataset_path'].split('hassaku/', 1)
        best_run_config['dataset_path'] = os.path.join(preamble_path, 'hassaku', post)
        pre, post = best_run_config['data_path'].split('hassaku/', 1)
        best_run_config['data_path'] = os.path.join(preamble_path, 'hassaku', post)

    # Running from non-main folder
    best_run_config['model_save_path'] = os.path.join(project_base_directory, best_run_config['model_save_path'])
    best_run_config['model_path'] = os.path.join(project_base_directory, best_run_config['model_path'])
    return best_run_config


def build_user_and_item_tag_matrix(path_to_dataset_folder: str = './data/ml1m', alpha_smoothening: float = .01):
    """
    Builds the user x tag matrix and the item x tag matrix on the training data. For the user x tag matrix, each row
    represents the tag frequencies in that user train data. N.B. As multiple genres/tags can appear in an item, we
    perform row-wise normalization across the item-tag matrix **before** constructing the user-tag matrix. E.g. when a
    user watches a Western movie then their propensity (~in frequency) towards Western movies is increased by 1. When a
    user watches a Western|Sci-Fi movie then their propensity is split by both genres, effectively increasing 0.5 for
    Western and 0.5 for Sci-Fi. This procedure is equivalent to Harald Steck "Calibrated Recommendations" RecSys 2018.
    @param path_to_dataset_folder: Path to the dataset folder. Code will automatically fill out the rest,
    @param alpha_smoothening: alpha value used to smoothen the training distribution (eq. 7 in H. Steck Calibrated Recommendations)
    @return:
        - user_tag_matrix
        - item_tag_matrix
    """

    assert 0 <= alpha_smoothening <= 1, 'Alpha value out of bounds'

    # Load Tag Matrix & Training Data
    tag_csv = pd.read_csv(os.path.join(path_to_dataset_folder, 'processed_dataset/tag_idxs.csv'))
    item_tag_idxs_csv = pd.read_csv(os.path.join(path_to_dataset_folder, 'processed_dataset/item_tag_idxs.csv'))
    train_data = pd.read_csv(os.path.join(path_to_dataset_folder, 'processed_dataset/listening_history_train.csv'))[
        ['user_idx', 'item_idx']]

    n_tags = len(tag_csv)
    n_items = train_data.item_idx.nunique()
    n_users = train_data.user_idx.nunique()

    # Building Tag Matrix
    tag_matrix = torch.zeros(size=(n_items, n_tags), dtype=torch.float)
    tag_matrix[[item_tag_idxs_csv.item_idx, item_tag_idxs_csv.tag_idx]] = 1.

    # Normalizing row-wise
    tag_matrix /= tag_matrix.sum(-1)[:, None]

    # Building Train Matrix
    train_mtx = scipy.sparse.csr_matrix(
        (torch.ones(len(train_data), dtype=torch.int16), (train_data.user_idx, train_data.item_idx)),
        shape=(n_users, n_items)
    )

    # Computing User-Tag Frequencies
    users_tag_frequencies = train_mtx @ tag_matrix
    n_items_per_user = train_mtx.sum(-1).A
    users_tag_frequencies /= n_items_per_user

    # Smoothening (eq.7)
    users_tag_frequencies = alpha_smoothening / n_tags + (1 - alpha_smoothening) * users_tag_frequencies

    return torch.tensor(users_tag_frequencies), tag_matrix


def build_user_and_item_pop_matrix(path_to_dataset_folder: str = './data/ml1m', alpha_smoothening: float = .01):
    """
    Builds the user x pop matrix and the item x pop matrix on the training data.
    @param path_to_dataset_folder: Path to the dataset folder. Code will automatically fill out the rest,
    @param alpha_smoothening: alpha value used to smoothen the training distribution (eq. 7 in H. Steck Calibrated Recommendations)
    @return:
        - user_pop_matrix
        - item_pop_matrix
    """

    assert 0 <= alpha_smoothening <= 1, 'Alpha value out of bounds'

    # Load Training Data
    train_data = pd.read_csv(os.path.join(path_to_dataset_folder, 'processed_dataset/listening_history_train.csv'))[
        ['user_idx', 'item_idx']]

    n_items = train_data.item_idx.nunique()
    n_users = train_data.user_idx.nunique()

    train_mtx = scipy.sparse.csr_matrix(
        (torch.ones(len(train_data), dtype=torch.float), (train_data.user_idx, train_data.item_idx)),
        shape=(n_users, n_items))

    # --- Compute Item Popularity Matrix --- #
    items_pop = train_mtx.sum(0).A1  # [n_items]
    pop_mass = items_pop.sum()  # total number of interactions
    items_pop /= pop_mass  # Normalizing respect to the mass

    sorted_items_idxs = np.argsort(-items_pop)

    mtx_row_idx = []
    mtx_col_idx = []

    curr_pop_mass = 0

    end_top_threshold = 0.2
    end_middle_threshold = 0.8

    for item_idx in sorted_items_idxs:

        curr_pop_mass += items_pop[item_idx]
        mtx_row_idx.append(item_idx)

        if curr_pop_mass < end_top_threshold:
            mtx_col_idx.append(0)
        elif curr_pop_mass < end_middle_threshold:
            mtx_col_idx.append(1)
        else:
            mtx_col_idx.append(2)

    # Creating the Matrix
    items_pop_mtx = scipy.sparse.csr_matrix(
        (torch.ones(len(mtx_row_idx), dtype=torch.float), (mtx_row_idx, mtx_col_idx)),
        shape=(n_items, 3))

    # --- Computing User Popularity --- #

    user_pop_mtx = train_mtx @ items_pop_mtx
    user_pop_mtx = user_pop_mtx.A
    user_pop_mtx /= user_pop_mtx.sum(-1)[:, None]

    # Smoothening (eq.7)
    user_pop_mtx = alpha_smoothening / 3 + (1 - alpha_smoothening) * user_pop_mtx

    return torch.tensor(user_pop_mtx), torch.tensor(items_pop_mtx.A)


class FullEvaluatorCalibrationDecorator(FullEvaluator):
    """
    Class that decorates a FullEvaluator object to add calibration metrics.
    """

    CALIBRATION_K_VALUES = [10, 20, 50, 100]

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

    def get_results(self):
        return self.full_evaluator.get_results()
