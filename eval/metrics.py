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
