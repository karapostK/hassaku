import logging
import math
from typing import Union, Tuple, Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data

from algorithms.base_classes import SGDBasedRecommenderAlgorithm, RecommenderAlgorithm
from explanations.utils import protomf_post_val, protomfs_post_val
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


class ACF(SGDBasedRecommenderAlgorithm):
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
        return protomf_post_val(
            self.anchors,
            self.item_embed.weight,
            compute_cosine_sim,
            lambda x: 1 - x,
            "Items",
            curr_epoch)


class UProtoMF(SGDBasedRecommenderAlgorithm):
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

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return UProtoMF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_prototypes'],
                        conf['sim_proto_weight'], conf['sim_batch_weight'])

    def post_val(self, curr_epoch: int):
        return protomf_post_val(
            self.prototypes,
            self.user_embed.weight,
            compute_shifted_cosine_sim,
            lambda x: 2 - x,
            "Users",
            curr_epoch)


class IProtoMF(SGDBasedRecommenderAlgorithm):
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
        return protomf_post_val(
            self.prototypes,
            self.item_embed.weight,
            compute_shifted_cosine_sim,
            lambda x: 2 - x,
            "Items",
            curr_epoch)


class UIProtoMF(SGDBasedRecommenderAlgorithm):
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


class UProtoMFs(SGDBasedRecommenderAlgorithm):
    """
    Implements a slightly simplified ProtoMF model with user prototypes. It differs from the original ProtoMF on:
        - No regularization losses are enforced
        - Entity-to-Prototype similarities can be negative (full-cosine similarity).
        - Other-entity (in this case items) weights are constrained to be positive. (Using a RelU)
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, n_prototypes: int = 20):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes

        self.user_embed = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embed = nn.Embedding(self.n_items, self.n_prototypes)
        self.relu = nn.ReLU()
        self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]) * .1 / self.embedding_dim,
                                       requires_grad=True)

        self.user_embed.apply(general_weight_init)
        torch.nn.init.trunc_normal_(self.item_embed.weight, mean=0.5, std=.1 / self.embedding_dim, a=0, b=1)

        self.name = 'UProtoMFs'

        logging.info(f'Built {self.name} model \n'
                     f'- n_users: {self.n_users} \n'
                     f'- n_items: {self.n_items} \n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- n_prototypes: {self.n_prototypes} \n')

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_embed = self.user_embed(u_idxs)
        sim_mtx = compute_cosine_sim(u_embed, self.prototypes)  # [batch_size, n_prototypes]

        return sim_mtx

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.relu(self.item_embed(i_idxs))  # [batch_size, n_neg + 1, n_prototypes]

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        dots = (u_repr.unsqueeze(-2) * i_repr).sum(dim=-1)
        return dots

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return UProtoMFs(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_prototypes'])

    def post_val(self, curr_epoch: int):
        return protomfs_post_val(
            self.prototypes,
            self.user_embed.weight,
            self.relu(self.item_embed.weight),
            compute_cosine_sim,
            lambda x: 1 - x,
            "Users",
            curr_epoch)


class IProtoMFs(SGDBasedRecommenderAlgorithm):
    """
    Implements a slightly simplified ProtoMF model with item prototypes. It differs from the original ProtoMF on:
        - No regularization losses are enforced
        - Entity-to-Prototype similarities can be negative (full-cosine similarity).
        - Other-entity (in this case items) weights are constrained to be positive. (Using a RelU)
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, n_prototypes: int = 20):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes

        self.user_embed = nn.Embedding(self.n_users, self.n_prototypes)
        self.item_embed = nn.Embedding(self.n_items, self.embedding_dim)
        self.relu = nn.ReLU()
        self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]) * .1 / self.embedding_dim,
                                       requires_grad=True)

        self.item_embed.apply(general_weight_init)
        torch.nn.init.trunc_normal_(self.user_embed.weight, mean=0.5, std=.1 / self.embedding_dim, a=0, b=1)

        self.name = 'IProtoMFs'

        logging.info(f'Built {self.name} model \n'
                     f'- n_users: {self.n_users} \n'
                     f'- n_items: {self.n_items} \n'
                     f'- embedding_dim: {self.embedding_dim} \n'
                     f'- n_prototypes: {self.n_prototypes} \n')

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.relu(self.user_embed(u_idxs))  # [batch_size, n_prototypes]

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_embed = self.item_embed(i_idxs)
        i_embed = i_embed.reshape(-1, i_embed.shape[-1])
        sim_mtx = compute_cosine_sim(i_embed, self.prototypes)
        sim_mtx = sim_mtx.reshape(list(i_idxs.shape) + [self.n_prototypes])
        return sim_mtx

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        dots = (u_repr.unsqueeze(-2) * i_repr).sum(dim=-1)
        return dots

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return IProtoMFs(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_prototypes'])

    def post_val(self, curr_epoch: int):
        return protomfs_post_val(
            self.prototypes,
            self.item_embed.weight,
            self.relu(self.user_embed.weight),
            compute_cosine_sim,
            lambda x: 1 - x,
            "Items",
            curr_epoch)


class UIProtoMFs(SGDBasedRecommenderAlgorithm):
    """
    Implements a slightly simplified ProtoMF model with user and item prototypes.
    It differs from the original ProtoMF on:
        - No regularization losses are enforced
        - Entity-to-Prototype similarities can be negative (full-cosine similarity).
        - Other-entity (in this case items) weights are constrained to be positive. (Using a RelU)
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, u_n_prototypes: int = 20,
                 i_n_prototypes: int = 20):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.uprotomfs = UProtoMFs(n_users, n_items, embedding_dim, u_n_prototypes)

        self.iprotomfs = IProtoMFs(n_users, n_items, embedding_dim, i_n_prototypes)

        self.u_to_i_proj = nn.Linear(self.embedding_dim, i_n_prototypes, bias=False)  # UProtoMFs -> IProtoMFs
        self.i_to_u_proj = nn.Linear(self.embedding_dim, u_n_prototypes, bias=False)  # IProtoMFs -> UProtoMFs

        self.relu = nn.ReLU()

        self.u_to_i_proj.apply(general_weight_init)
        self.i_to_u_proj.apply(general_weight_init)

        # Deleting unused parameters

        del self.uprotomfs.item_embed
        del self.iprotomfs.user_embed

        self.name = 'UIProtoMFs'

        logging.info(f'Built {self.name} model \n')

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_sim_mtx = self.uprotomfs.get_user_representations(u_idxs)
        u_proj = self.relu(self.u_to_i_proj(self.uprotomfs.user_embed(u_idxs)))

        return u_sim_mtx, u_proj

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_sim_mtx = self.iprotomfs.get_item_representations(i_idxs)
        i_proj = self.relu(self.i_to_u_proj(self.iprotomfs.item_embed(i_idxs)))

        return i_sim_mtx, i_proj

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        u_sim_mtx, u_proj = u_repr
        i_sim_mtx, i_proj = i_repr

        u_dots = (u_sim_mtx.unsqueeze(-2) * i_proj).sum(dim=-1)
        i_dots = (u_proj.unsqueeze(-2) * i_sim_mtx).sum(dim=-1)
        dots = u_dots + i_dots
        return dots

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return UIProtoMFs(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['u_n_prototypes'],
                          conf['i_n_prototypes'])

    def post_val(self, curr_epoch: int):
        uprotomfs_post_val = protomfs_post_val(self.uprotomfs.prototypes,
                                               self.uprotomfs.user_embed.weight,
                                               self.relu(self.i_to_u_proj(self.iprotomfs.item_embed.weight)),
                                               compute_cosine_sim,
                                               lambda x: (1 - x) / 2,
                                               "Users",
                                               curr_epoch)
        iprotomfs_post_val = protomfs_post_val(self.iprotomfs.prototypes,
                                               self.iprotomfs.item_embed.weight,
                                               self.relu(self.u_to_i_proj(self.uprotomfs.user_embed.weight)),
                                               compute_cosine_sim,
                                               lambda x: (1 - x) / 2,
                                               "Items",
                                               curr_epoch)
        uprotomfs_post_val = {'user_' + k: v for k, v in uprotomfs_post_val.items()}
        iprotomfs_post_val = {'item_' + k: v for k, v in iprotomfs_post_val.items()}
        return {**uprotomfs_post_val, **iprotomfs_post_val}


class UIProtoMFsCombine(RecommenderAlgorithm):
    """
    It encases UProtoMFs and IProtoMFs. Make sure that the models weights are loaded before calling __init__.
    No optimization is needed.
    """

    def save_model_to_path(self, path: str):
        raise ValueError(
            'This class cannot be saved to path since it made of 2 separate models (that should have been already saved'
            ' somewhere). Save the UProtoMFs and IProtoMFs models separately. If you want to optimize a UIProtoMF model,'
            ' use the UIProtoMF/s classes.')

    def load_model_from_path(self, path: str):
        raise ValueError(
            'This class cannot be loaded from path since it made of 2 separate models (that should have been already loaded'
            'from somewhere). Load the UProtoMFs and IProtoMFs models separately. If you want to optimize a UIProtoMF model,'
            ' use the UIProtoMF/s classes.')

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        raise ValueError(
            'This class cannot be built from conf since it made of 2 separate models. If you want to optimize a UIProtoMF model,'
            ' use the UIProtoMF/s classes.')

    def __init__(self, uprotomfs: UProtoMFs, iprotomfs: IProtoMFs):
        super().__init__()

        self.uprotomfs = uprotomfs
        self.iprotomfs = iprotomfs

        self.name = 'UIProtoMFsCombine'

        logging.info(f'Built {self.name} model \n')

    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        return self.uprotomfs.predict(u_idxs, i_idxs) + self.iprotomfs.predict(u_idxs, i_idxs)
