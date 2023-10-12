import pandas as pd
import torch
from torch import Tensor

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

    if in_idxs is None and only_idxs is None:
        entry_mask = torch.ones(in_repr.shape[0], dtype=bool).to(in_repr.device)
    elif in_idxs is not None and only_idxs is not None:
        entry_mask = torch.isin(only_idxs, in_idxs, assume_unique=True)
    else:
        raise ValueError(
            'Having one of in_indxs and only_indxs None while the other is set is not a valid combination!')

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
