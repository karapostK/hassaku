import inspect
import logging
import math
import os
from typing import Union, Tuple, Dict, List

import torch
from scipy import sparse as sp
from torch import nn
from torch.nn import functional as F
from torch.utils import data

from algorithms.base_classes import SGDBasedRecommenderAlgorithm, PrototypeWrapper
from explanations.utils import protomf_post_val_light
from train.utils import general_weight_init


def compute_norm_cosine_sim(x: torch.Tensor, y: torch.Tensor):
    """
    Computes the normalized shifted cosine similarity between two tensors.
    x and y have the same last dimension.
    """
    x_norm = F.normalize(x)
    y_norm = F.normalize(y)

    sim_mtx = (1 + x_norm @ y_norm.T) / 2
    sim_mtx = torch.clamp(sim_mtx, min=0., max=1.)

    return sim_mtx


def compute_shifted_cosine_sim(x: torch.Tensor, y: torch.Tensor):
    """
    Computes the shifted cosine similarity between two tensors.
    x and y have the same last dimension.
    """
    x_norm = F.normalize(x)
    y_norm = F.normalize(y)

    sim_mtx = (1 + x_norm @ y_norm.T)
    sim_mtx = torch.clamp(sim_mtx, min=0., max=2.)

    return sim_mtx


def compute_cosine_sim(x: torch.Tensor, y: torch.Tensor):
    """
    Computes the cosine similarity between two tensors.
    x and y have the same last dimension.
    """
    x_norm = F.normalize(x)
    y_norm = F.normalize(y)

    sim_mtx = x_norm @ y_norm.T
    sim_mtx = torch.clamp(sim_mtx, min=-1., max=1.)

    return sim_mtx


def entropy_from_softmax(p: torch.Tensor, p_unnorm: torch.Tensor):
    """
    Computes the entropy of a probability distribution assuming the distribution was obtained by softmax. It uses the
    un-normalized probabilities for numerical stability.
    @param p: tensor containing the probability of events xs. Shape is [*, n_events]
    @param p_unnorm: tensor contained the un-normalized probabilities (logits) of events xs. Shape is [*, n_events]
    @return: entropy of p. Shape is [*]
    """

    return (- (p * (p_unnorm - torch.logsumexp(p_unnorm, dim=-1, keepdim=True)))).sum(-1)


class SGDBaseline(SGDBasedRecommenderAlgorithm):
    """
    Implements a simple baseline comprised of biases (global, user, and item).
    See https://dl.acm.org/doi/10.1145/1401890.1401944
    """

    def __init__(self, n_users: int, n_items: int):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items

        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.apply(general_weight_init)

        self.name = 'SGDBaseline'

        logging.info(f'Built {self.name} module\n')

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.user_bias(u_idxs)

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.item_bias(i_idxs).squeeze()

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        out = u_repr + i_repr + self.global_bias
        return out

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        return SGDBaseline(dataset.n_users, dataset.n_items)


class SGDMatrixFactorization(SGDBasedRecommenderAlgorithm):
    """
    Implements a simple Matrix Factorization model trained with gradient descent
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, use_user_bias: bool = False,
                 use_item_bias: bool = False, use_global_bias: bool = False):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.use_user_bias = use_user_bias
        self.use_item_bias = use_item_bias
        self.use_global_bias = use_global_bias

        self.user_embeddings = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embeddings = nn.Embedding(self.n_items, self.embedding_dim)

        if self.use_user_bias:
            self.user_bias = nn.Embedding(self.n_users, 1)
        if self.use_item_bias:
            self.item_bias = nn.Embedding(self.n_items, 1)

        self.apply(general_weight_init)

        if self.use_global_bias:
            self.global_bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.name = 'SGDMatrixFactorization'

        logging.info(f'Built {self.name} module\n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- use_user_bias: {self.use_user_bias} \n'
                     f'- use_item_bias: {self.use_item_bias} \n'
                     f'- use_global_bias: {self.use_global_bias}')

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if self.use_user_bias:
            return self.user_embeddings(u_idxs), self.user_bias(u_idxs)
        else:
            return self.user_embeddings(u_idxs)

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if self.use_item_bias:
            return self.item_embeddings(i_idxs), self.item_bias(i_idxs).squeeze()
        return self.item_embeddings(i_idxs)

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        if isinstance(u_repr, tuple):
            u_embed, u_bias = u_repr
        else:
            u_embed = u_repr

        if isinstance(i_repr, tuple):
            i_embed, i_bias = i_repr
        else:
            i_embed = i_repr

        out = (u_embed[:, None, :] * i_embed).sum(dim=-1)

        if self.use_user_bias:
            out += u_bias
        if self.use_item_bias:
            out += i_bias
        if self.use_global_bias:
            out += self.global_bias
        return out

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        return SGDMatrixFactorization(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['use_user_bias'],
                                      conf['use_item_bias'], conf['use_global_bias'])


class ACF(PrototypeWrapper):
    """
    Implements Anchor-based Collaborative Filtering by Barkan et al.(https://dl.acm.org/doi/pdf/10.1145/3459637.3482056)
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, n_anchors: int = 20,
                 delta_exc: float = 1e-1, delta_inc: float = 1e-2):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_anchors = n_anchors
        self.delta_exc = delta_exc
        self.delta_inc = delta_inc

        # NB. In order to ensure stability, ACF's weights **need** not to be initialized with small values.
        self.anchors = nn.Parameter(torch.randn([self.n_anchors, self.embedding_dim]), requires_grad=True)

        self.user_embed = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embed = nn.Embedding(self.n_items, self.embedding_dim)

        self._acc_exc = 0
        self._acc_inc = 0

        self.name = 'ACF'

        logging.info(f'Built {self.name} model \n'
                     f'- n_users: {self.n_users} \n'
                     f'- n_items: {self.n_items} \n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- n_anchors: {self.n_anchors} \n'
                     f'- delta_exc: {self.delta_exc} \n'
                     f'- delta_inc: {self.delta_inc} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        dots = self.combine_user_item_representations(u_repr, i_repr)

        _, c_i, c_i_unnorm = i_repr

        # Exclusiveness constraint
        exc_values = entropy_from_softmax(c_i, c_i_unnorm)  # [batch_size, n_neg +1] or [batch_size]
        exc_loss = exc_values.mean()

        # Inclusiveness constraint
        c_i_flat = c_i.reshape(-1, self.n_anchors)  # [*, n_anchors]
        q_k = c_i_flat.sum(dim=0) / c_i.sum()  # [n_anchors]
        inc_entropy = (- q_k * torch.log(q_k)).sum()
        inc_loss = math.log(self.n_anchors) - inc_entropy  # Maximizing the Entropy

        self._acc_exc += exc_loss
        self._acc_inc += inc_loss

        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_embed = self.user_embed(u_idxs)  # [batch_size, embedding_dim]
        c_u = u_embed @ self.anchors.T  # [batch_size, n_anchors]
        c_u = nn.Softmax(dim=-1)(c_u)

        u_anc = c_u @ self.anchors  # [batch_size, embedding_dim]

        return u_anc

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_embed = self.item_embed(i_idxs)  # [batch_size, (n_neg + 1), embedding_dim]
        c_i_unnorm = i_embed @ self.anchors.T  # [batch_size, (n_neg + 1), n_anchors]
        c_i = nn.Softmax(dim=-1)(c_i_unnorm)  # [batch_size, (n_neg + 1), n_anchors]

        i_anc = c_i @ self.anchors  # [batch_size, (n_neg + 1), embedding_dim]
        return i_anc, c_i, c_i_unnorm

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        u_anc = u_repr
        i_anc = i_repr[0]
        dots = (u_anc.unsqueeze(-2) * i_anc).sum(dim=-1)
        return dots

    def get_item_representations_pre_tune(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_embed = self.item_embed(i_idxs)  # [batch_size, (n_neg + 1), embedding_dim]
        c_i_unnorm = i_embed @ self.anchors.T  # [batch_size, (n_neg + 1), n_anchors]
        c_i = nn.Softmax(dim=-1)(c_i_unnorm)  # [batch_size, (n_neg + 1), n_anchors]
        return c_i

    def get_item_representations_post_tune(self, c_i: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_anc = c_i @ self.anchors  # [batch_size, (n_neg + 1), embedding_dim]
        return i_anc, c_i, None

    def get_user_representations_pre_tune(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_embed = self.user_embed(u_idxs)  # [batch_size, embedding_dim]
        c_u = u_embed @ self.anchors.T  # [batch_size, n_anchors]
        c_u = nn.Softmax(dim=-1)(c_u)
        return c_u

    def get_user_representations_post_tune(self, c_u: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_anc = c_u @ self.anchors  # [batch_size, embedding_dim]
        return u_anc

    def get_and_reset_other_loss(self) -> Dict:
        _acc_exc, _acc_inc = self._acc_exc, self._acc_inc
        self._acc_exc = self._acc_inc = 0
        exc_loss = self.delta_exc * _acc_exc
        inc_loss = self.delta_inc * _acc_inc

        return {
            'reg_loss': exc_loss + inc_loss,
            'exc_loss': exc_loss,
            'inc_loss': inc_loss
        }

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return ACF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_anchors'],
                   conf['delta_exc'], conf['delta_inc'])

    def post_val(self, curr_epoch: int):
        return protomf_post_val_light(
            self.anchors,
            self.item_embed.weight,
            compute_cosine_sim,
            lambda x: 1 - x,
            "Items",
            curr_epoch)


class UProtoMF(PrototypeWrapper):
    """
    Implements the ProtoMF model with user prototypes as defined in https://dl.acm.org/doi/abs/10.1145/3523227.3546756
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, n_prototypes: int = 20,
                 sim_proto_weight: float = 1., sim_batch_weight: float = 1.):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes
        self.sim_proto_weight = sim_proto_weight
        self.sim_batch_weight = sim_batch_weight

        self.user_embed = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embed = nn.Embedding(self.n_items, self.n_prototypes)

        self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]) * .1 / self.embedding_dim,
                                       requires_grad=True)

        self.user_embed.apply(general_weight_init)
        self.item_embed.apply(general_weight_init)

        self._acc_r_proto = 0
        self._acc_r_batch = 0

        self.name = 'UProtoMF'

        logging.info(f'Built {self.name} model \n'
                     f'- n_users: {self.n_users} \n'
                     f'- n_items: {self.n_items} \n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- n_prototypes: {self.n_prototypes} \n'
                     f'- sim_proto_weight: {self.sim_proto_weight} \n'
                     f'- sim_batch_weight: {self.sim_batch_weight} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        dots = self.combine_user_item_representations(u_repr, i_repr)

        self.compute_reg_losses(u_repr)

        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_embed = self.user_embed(u_idxs)
        sim_mtx = compute_shifted_cosine_sim(u_embed, self.prototypes)  # [batch_size, n_prototypes]

        return sim_mtx

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.item_embed(i_idxs)

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        dots = (u_repr.unsqueeze(-2) * i_repr).sum(dim=-1)
        return dots

    def compute_reg_losses(self, sim_mtx):
        # Compute regularization losses
        sim_mtx = sim_mtx.reshape(-1, self.n_prototypes)
        dis_mtx = (2 - sim_mtx)  # Equivalent to maximizing the similarity.
        self._acc_r_proto += dis_mtx.min(dim=0).values.mean()
        self._acc_r_batch += dis_mtx.min(dim=1).values.mean()

    def get_and_reset_other_loss(self) -> Dict:
        acc_r_proto, acc_r_batch = self._acc_r_proto, self._acc_r_batch
        self._acc_r_proto = self._acc_r_batch = 0
        proto_loss = self.sim_proto_weight * acc_r_proto
        batch_loss = self.sim_batch_weight * acc_r_batch
        return {
            'reg_loss': proto_loss + batch_loss,
            'proto_loss': proto_loss,
            'batch_loss': batch_loss
        }

    def get_user_representations_pre_tune(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_repr = self.get_user_representations(u_idxs)
        return u_repr

    def get_user_representations_post_tune(self, u_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return u_repr

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return UProtoMF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_prototypes'],
                        conf['sim_proto_weight'], conf['sim_batch_weight'])

    def post_val(self, curr_epoch: int):
        return protomf_post_val_light(
            self.prototypes,
            self.user_embed.weight,
            compute_shifted_cosine_sim,
            lambda x: 2 - x,
            "Users",
            curr_epoch)


class IProtoMF(PrototypeWrapper):
    """
    Implements the ProtoMF model with item prototypes as defined in https://dl.acm.org/doi/abs/10.1145/3523227.3546756
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, n_prototypes: int = 20,
                 sim_proto_weight: float = 1., sim_batch_weight: float = 1.):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes
        self.sim_proto_weight = sim_proto_weight
        self.sim_batch_weight = sim_batch_weight

        self.user_embed = nn.Embedding(self.n_users, self.n_prototypes)
        self.item_embed = nn.Embedding(self.n_items, self.embedding_dim)

        self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]) * .1 / self.embedding_dim,
                                       requires_grad=True)

        self.user_embed.apply(general_weight_init)
        self.item_embed.apply(general_weight_init)

        self._acc_r_proto = 0
        self._acc_r_batch = 0

        self.name = 'IProtoMF'

        logging.info(f'Built {self.name} model \n'
                     f'- n_users: {self.n_users} \n'
                     f'- n_items: {self.n_items} \n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- n_prototypes: {self.n_prototypes} \n'
                     f'- sim_proto_weight: {self.sim_proto_weight} \n'
                     f'- sim_batch_weight: {self.sim_batch_weight} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        dots = self.combine_user_item_representations(u_repr, i_repr)

        self.compute_reg_losses(i_repr)

        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.user_embed(u_idxs)

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_embed = self.item_embed(i_idxs)
        i_embed = i_embed.reshape(-1, i_embed.shape[-1])
        sim_mtx = compute_shifted_cosine_sim(i_embed, self.prototypes)
        sim_mtx = sim_mtx.reshape(list(i_idxs.shape) + [self.n_prototypes])

        return sim_mtx

    def get_item_representations_pre_tune(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_repr = self.get_item_representations(i_idxs)
        return i_repr

    def get_item_representations_post_tune(self, i_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return i_repr

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        dots = (u_repr.unsqueeze(-2) * i_repr).sum(dim=-1)
        return dots

    def compute_reg_losses(self, sim_mtx):
        # Compute regularization losses
        sim_mtx = sim_mtx.reshape(-1, self.n_prototypes)
        dis_mtx = (2 - sim_mtx)  # Equivalent to maximizing the similarity.
        self._acc_r_proto += dis_mtx.min(dim=0).values.mean()
        self._acc_r_batch += dis_mtx.min(dim=1).values.mean()

    def get_and_reset_other_loss(self) -> Dict:
        acc_r_proto, acc_r_batch = self._acc_r_proto, self._acc_r_batch
        self._acc_r_proto = self._acc_r_batch = 0
        proto_loss = self.sim_proto_weight * acc_r_proto
        batch_loss = self.sim_batch_weight * acc_r_batch
        return {
            'reg_loss': proto_loss + batch_loss,
            'proto_loss': proto_loss,
            'batch_loss': batch_loss
        }

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return IProtoMF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_prototypes'],
                        conf['sim_proto_weight'], conf['sim_batch_weight'])

    def post_val(self, curr_epoch: int):
        return protomf_post_val_light(
            self.prototypes,
            self.item_embed.weight,
            compute_shifted_cosine_sim,
            lambda x: 2 - x,
            "Items",
            curr_epoch)


class UIProtoMF(PrototypeWrapper):
    """
    Implements the ProtoMF model with item and user prototypes as defined in https://dl.acm.org/doi/abs/10.1145/3523227.3546756
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, u_n_prototypes: int = 20,
                 i_n_prototypes: int = 20, u_sim_proto_weight: float = 1., u_sim_batch_weight: float = 1.,
                 i_sim_proto_weight: float = 1., i_sim_batch_weight: float = 1.):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.uprotomf = UProtoMF(n_users, n_items, embedding_dim, u_n_prototypes,
                                 u_sim_proto_weight, u_sim_batch_weight)

        self.iprotomf = IProtoMF(n_users, n_items, embedding_dim, i_n_prototypes,
                                 i_sim_proto_weight, i_sim_batch_weight)

        self.u_to_i_proj = nn.Linear(self.embedding_dim, i_n_prototypes, bias=False)  # UProtoMF -> IProtoMF
        self.i_to_u_proj = nn.Linear(self.embedding_dim, u_n_prototypes, bias=False)  # IProtoMF -> UProtoMF

        self.u_to_i_proj.apply(general_weight_init)
        self.i_to_u_proj.apply(general_weight_init)

        # Deleting unused parameters

        del self.uprotomf.item_embed
        del self.iprotomf.user_embed

        self.name = 'UIProtoMF'

        logging.info(f'Built {self.name} model \n')

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_sim_mtx = self.uprotomf.get_user_representations(u_idxs)
        u_proj = self.u_to_i_proj(self.uprotomf.user_embed(u_idxs))

        return u_sim_mtx, u_proj

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_sim_mtx = self.iprotomf.get_item_representations(i_idxs)
        i_proj = self.i_to_u_proj(self.iprotomf.item_embed(i_idxs))

        return i_sim_mtx, i_proj

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        u_sim_mtx, u_proj = u_repr
        i_sim_mtx, i_proj = i_repr

        u_dots = (u_sim_mtx.unsqueeze(-2) * i_proj).sum(dim=-1)
        i_dots = (u_proj.unsqueeze(-2) * i_sim_mtx).sum(dim=-1)
        dots = u_dots + i_dots
        return dots

    def get_item_representations_pre_tune(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_repr = self.get_item_representations(i_idxs)
        return i_repr

    def get_item_representations_post_tune(self, i_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return i_repr

    def get_user_representations_pre_tune(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_repr = self.get_user_representations(u_idxs)
        return u_repr

    def get_user_representations_post_tune(self, u_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return u_repr

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        dots = self.combine_user_item_representations(u_repr, i_repr)

        u_sim_mtx, _ = u_repr
        i_sim_mtx, _ = i_repr
        self.uprotomf.compute_reg_losses(u_sim_mtx)
        self.iprotomf.compute_reg_losses(i_sim_mtx)

        return dots

    def get_and_reset_other_loss(self) -> Dict:
        u_reg = {'user_' + k: v for k, v in self.uprotomf.get_and_reset_other_loss().items()}
        i_reg = {'item_' + k: v for k, v in self.iprotomf.get_and_reset_other_loss().items()}
        return {
            'reg_loss': u_reg.pop('user_reg_loss') + i_reg.pop('item_reg_loss'),
            **u_reg,
            **i_reg
        }

    def post_val(self, curr_epoch: int):
        uprotomf_post_val = {'user_' + k: v for k, v in self.uprotomf.post_val(curr_epoch).items()}
        iprotomf_post_val = {'item_' + k: v for k, v in self.iprotomf.post_val(curr_epoch).items()}
        return {**uprotomf_post_val, **iprotomf_post_val}

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return UIProtoMF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['u_n_prototypes'],
                         conf['i_n_prototypes'], conf['u_sim_proto_weight'], conf['u_sim_batch_weight'],
                         conf['i_sim_proto_weight'], conf['i_sim_batch_weight'])


class ECF(PrototypeWrapper):
    """
    Implements the ECF model from https://dl.acm.org/doi/10.1145/3543507.3583303
    """

    def __init__(self, n_users: int, n_items: int, tag_matrix: sp.csr_matrix, interaction_matrix: sp.csr_matrix,
                 embedding_dim: int = 100, n_clusters: int = 64, top_n: int = 20, top_m: int = 20,
                 temp_masking: float = 2., temp_tags: float = 2., top_p: int = 4, lam_cf: float = 0.6,
                 lam_ind: float = 1., lam_ts: float
                 = 1.):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.tag_matrix = nn.Parameter(torch.from_numpy(tag_matrix.A), requires_grad=False).float()
        self.interaction_matrix = nn.Parameter(torch.from_numpy(interaction_matrix.A), requires_grad=False).float()

        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.top_n = top_n
        self.top_m = top_m
        self.temp_masking = temp_masking
        self.temp_tags = temp_tags
        self.top_p = top_p

        self.lam_cf = lam_cf
        self.lam_ind = lam_ind
        self.lam_ts = lam_ts

        self.user_embed = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embed = nn.Embedding(self.n_items, self.embedding_dim)

        indxs = torch.randperm(self.n_items)[:self.n_clusters]
        self.clusters = nn.Parameter(self.item_embed.weight[indxs].detach(), requires_grad=True)

        self._acc_ts = 0
        self._acc_ind = 0
        self._acc_cf = 0

        # Parameters are set every batch
        self._x_tildes = None
        self._xs = None

        self.name = 'ECF'

        logging.info(f'Built {self.name} model \n'
                     f'- n_users: {self.n_users} \n'
                     f'- n_items: {self.n_items} \n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- n_clusters: {self.n_clusters} \n'
                     f'- lam_cf: {self.lam_cf} \n'
                     f'- top_n: {self.top_n} \n'
                     f'- top_m: {self.top_m} \n'
                     f'- temp_masking: {self.temp_masking} \n'
                     f'- temp_tags: {self.temp_tags} \n'
                     f'- top_p: {self.top_p} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        i_repr = self.get_item_representations(i_idxs)
        # NB. item representations should be generated before calling user_representations
        u_repr = self.get_user_representations(u_idxs)

        dots = self.combine_user_item_representations(u_repr, i_repr)

        # Tag Loss
        # N.B. Frequency weighting factor is already included in the tag_matrix.
        d_c = self._xs.T @ self.tag_matrix.to(u_idxs.device)  # [n_clusters, n_tags]
        # since log is a monotonic function we can invert the order between log and topk
        log_b_c = nn.LogSoftmax(-1)(d_c / self.temp_tags)
        top_log_b_c = log_b_c.topk(self.top_p, dim=-1).values  # [n_clusters, top_p]

        loss_tags = (- top_log_b_c).sum()
        self._acc_ts += loss_tags

        # Independence Loss
        sim_mtx = compute_cosine_sim(self.clusters, self.clusters)
        self_sim = torch.diag(- nn.LogSoftmax(dim=-1)(sim_mtx))

        self._acc_ind += self_sim.sum()

        # BPR Loss
        u_embed = u_repr[1]
        i_embed = i_repr[1]

        logits = (u_embed.unsqueeze(-2) * i_embed).sum(dim=-1)

        pos_logits = logits[:, 0].unsqueeze(1)  # [batch_size,1]
        neg_logits = logits[:, 1:]  # [batch_size,n_neg]

        diff_logits = (pos_logits - neg_logits).flatten()
        labels = torch.ones_like(diff_logits, device=diff_logits.device)

        bpr_loss = nn.BCEWithLogitsLoss()(diff_logits, labels)
        self._acc_cf += bpr_loss

        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        y_u = self.interaction_matrix[u_idxs]  # [batch_size, n_items]
        u_embed = self.user_embed(u_idxs)

        a_tilde = y_u @ self._x_tildes  # [batch_size, n_clusters]

        # Creating exact mask
        m = torch.zeros_like(a_tilde).to(a_tilde.device)
        a_tilde_tops = a_tilde.topk(self.top_n).indices
        dummy_column = torch.arange(a_tilde.shape[0])[:, None].to(a_tilde.device)
        m[dummy_column, a_tilde_tops] = True

        # Creating approximated mask
        m_tilde = nn.Softmax(dim=-1)(a_tilde / self.temp_masking)

        # Putting together the masks
        m_hat = m_tilde + (m - m_tilde).detach()

        # Building affiliation vector
        a_i = nn.Sigmoid()(a_tilde) * m_hat

        return a_i, u_embed

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        self._generate_item_representations()

        i_embed = self.item_embed(i_idxs)  # [batch_size, embed_dim] or [batch_size, n_neg + 1, embed_dim]

        x_i = self._xs[i_idxs]

        return x_i, i_embed

    def _generate_item_representations(self):
        i_embed = self.item_embed.weight  # [n_items, embed_d]
        self._x_tildes = compute_cosine_sim(i_embed, self.clusters)  # [n_items, n_clusters]

        # Creating exact mask
        m = torch.zeros_like(self._x_tildes).to(self._x_tildes.device)
        x_tilde_tops = self._x_tildes.topk(self.top_m).indices  # [n_items, top_m]
        dummy_column = torch.arange(self.n_items)[:, None].to(self._x_tildes.device)
        m[dummy_column, x_tilde_tops] = True

        # Creating approximated mask
        m_tilde = nn.Softmax(dim=-1)(self._x_tildes / self.temp_masking)  # [n_items, n_clusters]

        # Putting together the masks
        m_hat = m_tilde + (m - m_tilde).detach()

        # Building affiliation vector
        self._xs = nn.Sigmoid()(self._x_tildes) * m_hat

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        a_i, _ = u_repr
        x_i, _ = i_repr

        sparse_dots = (a_i.unsqueeze(-2) * x_i).sum(dim=-1)
        return sparse_dots

    def get_item_representations_pre_tune(self, i_idxs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # i_idxs is ignored
        i_embed = self.item_embed.weight  # [n_items, embed_d]
        self._x_tildes = compute_cosine_sim(i_embed, self.clusters)  # [n_items, n_clusters]

        # Creating exact mask
        m = torch.zeros_like(self._x_tildes).to(self._x_tildes.device)
        x_tilde_tops = self._x_tildes.topk(self.top_m).indices  # [n_items, top_m]
        dummy_column = torch.arange(self.n_items)[:, None].to(self._x_tildes.device)
        m[dummy_column, x_tilde_tops] = True

        # Creating approximated mask
        m_tilde = nn.Softmax(dim=-1)(self._x_tildes / self.temp_masking)  # [n_items, n_clusters]

        # Putting together the masks
        m_hat = m_tilde + (m - m_tilde).detach()

        # Building affiliation vector
        self._xs = nn.Sigmoid()(self._x_tildes) * m_hat
        return self._xs, self.item_embed.weight

    def get_item_representations_post_tune(self, i_repr: torch.Tensor) -> Union[
        torch.Tensor, Tuple[torch.Tensor, ...]]:

        return i_repr

    def get_user_representations_pre_tune(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        y_u = self.interaction_matrix[u_idxs]  # [batch_size, n_items]
        u_embed = self.user_embed(u_idxs)
        a_tilde = y_u @ self._x_tildes  # [batch_size, n_clusters]

        # Creating exact mask
        m = torch.zeros_like(a_tilde).to(a_tilde.device)
        a_tilde_tops = a_tilde.topk(self.top_n).indices
        dummy_column = torch.arange(a_tilde.shape[0])[:, None].to(a_tilde.device)
        m[dummy_column, a_tilde_tops] = True

        # Creating approximated mask
        m_tilde = nn.Softmax(dim=-1)(a_tilde / self.temp_masking)

        # Putting together the masks
        m_hat = m_tilde + (m - m_tilde).detach()

        # Building affiliation vector
        a_i = nn.Sigmoid()(a_tilde) * m_hat

        return a_i, u_embed

    def get_user_representations_post_tune(self, u_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return u_repr

    def get_and_reset_other_loss(self) -> Dict:
        acc_ts, acc_ind, acc_cf = self._acc_ts, self._acc_ind, self._acc_cf
        self._acc_ts = self._acc_ind = self._acc_cf = 0
        cf_loss = self.lam_cf * acc_cf
        ind_loss = self.lam_ind * acc_ind
        ts_loss = self.lam_ts * acc_ts

        return {
            'reg_loss': ts_loss + ind_loss + cf_loss,
            'cf_loss': cf_loss,
            'ind_loss': ind_loss,
            'ts_loss': ts_loss
        }

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        init_signature = inspect.signature(ECF.__init__)
        def_parameters = {k: v.default for k, v in init_signature.parameters.items() if
                          v.default is not inspect.Parameter.empty}
        parameters = {**def_parameters, **conf}

        return ECF(dataset.n_users, dataset.n_items, dataset.tag_matrix,
                   dataset.sampling_matrix, parameters['embedding_dim'],
                   parameters['n_clusters'], parameters['top_n'], parameters['top_m'],
                   parameters['temp_masking'], parameters['temp_tags'],
                   parameters['top_p'], parameters['lam_cf'], parameters['lam_ind'],
                   parameters['lam_ts']
                   )

    def to(self, *args, **kwargs):
        for arg in args:
            if type(arg) == torch.device or arg == 'cuda' or arg == 'cpu':
                self.tag_matrix = self.tag_matrix.to(arg)
                self.interaction_matrix = self.interaction_matrix.to(arg)
        return super().to(*args, **kwargs)

    def load_model_from_path(self, path: str):
        path = os.path.join(path, 'model.pth')
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=False)
        print('Model Loaded')


class DeepMatrixFactorization(SGDBasedRecommenderAlgorithm):
    """
    Deep Matrix Factorization Models for Recommender Systems by Xue et al. (https://www.ijcai.org/Proceedings/2017/0447.pdf)
    """

    def __init__(self, matrix: sp.spmatrix, u_mid_layers: Union[List[int], int],
                 i_mid_layers: Union[List[int], int],
                 final_dimension: int):
        """
        :param matrix: user x item sparse matrix
        :param u_mid_layers: list of integers representing the size of the middle layers on the user side
        :param i_mid_layers: list of integers representing the size of the middle layers on the item side
        :param final_dimension: last dimension of the layers for both user and item side
        """
        super().__init__()
        self.n_users, self.n_items = matrix.shape
        # from equation (13) of the original paper
        self.mu = 1.e-6

        self.final_dimension = final_dimension

        if isinstance(u_mid_layers, int):
            u_mid_layers = [u_mid_layers]
        if isinstance(i_mid_layers, int):
            i_mid_layers = [i_mid_layers]

        self.u_layers = [self.n_items] + u_mid_layers + [self.final_dimension]
        self.i_layers = [self.n_users] + i_mid_layers + [self.final_dimension]

        # Building the network for the user
        u_nn = []
        for i, (n_in, n_out) in enumerate(zip(self.u_layers[:-1], self.u_layers[1:])):
            u_nn.append(nn.Linear(n_in, n_out))

            if i != len(self.u_layers) - 2:
                u_nn.append(nn.ReLU())
        self.user_nn = nn.Sequential(*u_nn)

        # Building the network for the item
        i_nn = []
        for i, (n_in, n_out) in enumerate(zip(self.i_layers[:-1], self.i_layers[1:])):
            i_nn.append(nn.Linear(n_in, n_out))

            if i != len(self.i_layers) - 2:
                i_nn.append(nn.ReLU())
        self.item_nn = nn.Sequential(*i_nn)

        for user_param in self.user_nn.parameters():
            user_param.requires_grad = True

        for item_param in self.item_nn.parameters():
            item_param.requires_grad = True

        self.cosine_func = nn.CosineSimilarity(dim=-1)

        # Unfortunately, it seems that there is no CUDA support for sparse matrices..
        self.user_vectors = nn.Embedding.from_pretrained(torch.Tensor(matrix.todense()))
        self.item_vectors = nn.Embedding.from_pretrained(self.user_vectors.weight.T)

        # Initialization of the network
        self.user_nn.apply(general_weight_init)
        self.item_nn.apply(general_weight_init)

        self.name = 'DeepMatrixFactorization'

        logging.info(f'Built {self.name} module \n'
                     f'- u_layers: {self.u_layers} \n'
                     f'- i_layers: {self.i_layers} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor):

        # User pass
        u_vec = self.get_user_representations(u_idxs)
        # Item pass
        i_vec = self.get_item_representations(i_idxs)
        # Cosine
        sim = self.combine_user_item_representations(u_vec, i_vec)

        return sim

    @staticmethod
    def build_from_conf(conf: dict, train_dataset: data.Dataset):
        train_dataset = train_dataset.iteration_matrix
        return DeepMatrixFactorization(train_dataset, conf['u_mid_layers'], conf['i_mid_layers'],
                                       conf['final_dimension'])

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_vec = self.item_vectors(i_idxs)
        i_vec = self.item_nn(i_vec)
        return i_vec

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        u_vec = self.user_vectors(u_idxs)
        u_vec = self.user_nn(u_vec)

        return u_vec

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        sim = self.cosine_func(u_repr[:, None, :], i_repr)
        sim[sim < self.mu] = self.mu

        return sim
