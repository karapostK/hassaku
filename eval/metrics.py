import torch


def recall_at_k_batch(logits: torch.Tensor, y_true: torch.Tensor, k: int = 10, aggr_sum: bool = True,
                      idx_topk: torch.Tensor = None):
    """
    Recall at k.
    :param logits: Logits. Shape is (batch_size, *).
    :param y_true: the true prediction. Shape is (batch_size, *)
    :param k: cut-off value
    :param aggr_sum: whether we sum the values over the batch_size. Default to true.
    :param idx_topk: pre-computed top-k indexes (used to index both logits and y_true)

    :return: Recall@k. Shape is (batch_size,) if aggr_sum=False, otherwise returns a scalar.
    """

    if idx_topk is not None:
        assert idx_topk.shape[-1] == k, 'Top-k indexes have different "k" compared to the parameter function'
    else:
        idx_topk = logits.topk(k=k).indices

    indexing_column = torch.arange(logits.shape[0]).unsqueeze(-1)

    num = y_true[indexing_column, idx_topk].sum(dim=-1)
    den = y_true.sum(dim=-1)

    recall = num / den

    recall[torch.isnan(recall)] = .0  # Recall is set to 0 for users without items in val/test

    assert (recall >= 0).all() and (recall <= 1).all(), 'Recall value is out of bound!'

    if aggr_sum:
        return recall.sum()
    else:
        return recall


def precision_at_k_batch(logits: torch.Tensor, y_true: torch.Tensor, k: int = 10, aggr_sum: bool = True,
                         idx_topk: torch.Tensor = None):
    """
    Precision at k.
    :param logits: Logits. Shape is (batch_size, *).
    :param y_true: the true prediction. Shape is (batch_size, *)
    :param k: cut-off value
    :param aggr_sum: whether we sum the values over the batch_size. Default to true.
    :param idx_topk: pre-computed top-k indexes (used to index both logits and y_true)

    :return: Precision@k. Shape is (batch_size,) if aggr_sum=False, otherwise returns a scalar.
    """

    if idx_topk is not None:
        assert idx_topk.shape[-1] == k, 'Top-k indexes have different "k" compared to the parameter function'
    else:
        idx_topk = logits.topk(k=k).indices

    indexing_column = torch.arange(logits.shape[0]).unsqueeze(-1)

    num = y_true[indexing_column, idx_topk].sum(dim=-1)

    precision = num / k

    assert (precision >= 0).all() and (precision <= 1).all(), 'Precision value is out of bound!'
    if aggr_sum:
        return precision.sum()
    else:
        return precision


def ndcg_at_k_batch(logits: torch.Tensor, y_true: torch.Tensor, k: int = 10, aggr_sum: bool = True,
                    idx_topk: torch.Tensor = None):
    """
    Normalized Discount Cumulative Gain at k. This implementation considers binary relevance.
    :param logits: Logits. Shape is (batch_size, *).
    :param y_true: the true prediction. Shape is (batch_size, *)
    :param k: cut-off value
    :param aggr_sum: whether we sum the values over the batch_size. Default to true.
    :param idx_topk: pre-computed top-k indexes (used to index both logits and y_true)

    :return: NDCG@k. Shape is (batch_size,) if aggr_sum=False, otherwise returns a scalar.
    """

    if idx_topk is not None:
        assert idx_topk.shape[-1] == k, 'Top-k indexes have different "k" compared to the parameter function'
    else:
        idx_topk = logits.topk(k=k).indices

    indexing_column = torch.arange(logits.shape[0]).unsqueeze(-1)

    discount_template = 1. / torch.log2(torch.arange(2, k + 2).float())
    discount_template = discount_template.to(logits.device)

    DCG = (y_true[indexing_column, idx_topk] * discount_template).sum(-1)
    IDCG = (y_true.topk(k).values * discount_template).sum(-1)

    NDCG = DCG / IDCG

    NDCG[torch.isnan(NDCG)] = .0  # Recall is set to 0 for users without items in val/test

    NDCG = NDCG.clamp(max=1.)  # Avoiding issues with the precision.

    if aggr_sum:
        return NDCG.sum()
    else:
        return NDCG


def hellinger_distance(p: torch.Tensor, q: torch.Tensor):
    """
    Computes the Hellinger Distance between two probability distributions.
    It is assumed that both p and q have the same domain. The distance is symmetric.
    # https://en.wikipedia.org/wiki/Hellinger_distance
    @param p: First Probability Distribution. Shape is [*, d] where d is the discrete # of events
    @param q: Second Probability Distribution. Shape is the same as p.
    @return: Hellinger Distance. Shape is [*]
    """
    diff = torch.sqrt(p) - torch.sqrt(q)
    squared_diff = diff ** 2
    return torch.sqrt(.5 * squared_diff.sum(-1))


def kl_divergence(true_p: torch.Tensor, model_q: torch.Tensor):
    """
    Computes the Kullback-Leibler Divergence between two probability distribution. The divergence is NOT asymmetric.
    It is assumed that both p and q have the same domain.
    # https://dl.acm.org/doi/pdf/10.1145/3240323.3240372
    @param true_p: "represents the data, the observations, or a measured probability distribution" (from Wikipedia)
    @param model_q: "represents instead a theory, a model, a description or an approximation of P" (from Wikipedia)
    @return: The KL divergence of model_p from true_p.
    """
    return (true_p * (true_p.log() - model_q.log())).sum(-1)


def jensen_shannon_distance(p: torch.Tensor, q: torch.Tensor):
    """
    Computes the Jensen Shannon Distance between two probability distributions.
    It is assumed that both p and q have the same domain. The distance is symmetric.
    *NB.* The function will return nan if one of the two probability distribution is not defined on some event!
    To avoid this result, it is advised to blend each of the probability distribution with a uniform distribution over
    all the events. E.g. assuming that p is defined on 10 events: p = (1-α) * p + α * 1/d with α equal to a small value
    such as α= .01
    # https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    @param p: First Probability Distribution. Shape is [*, d] where d is the discrete # of events
    @param q: Second Probability Distribution. Shape is the same as p.
    @return: Jensen Shannon Distance. Shape is [*]
    """
    m = (.5 * (p + q))
    kl_p_m = kl_divergence(p, m)
    kl_q_m = kl_divergence(q, m)

    jsd = .5 * (kl_p_m + kl_q_m)
    return torch.sqrt(jsd)
