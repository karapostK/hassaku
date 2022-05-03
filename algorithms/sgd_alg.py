from typing import Union, Tuple

import torch
from torch import nn
from torch.utils import data

from algorithms.base_classes import SGDBasedRecommenderAlgorithm
from train.utils import general_weight_init


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

        print(f'Built {self.name} module\n')

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

        print(f'Built {self.name} module\n'
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
    NB. Loss aggregation has to be performed differently in order to have the regularization losses in the same size
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

        self.anchors = nn.Parameter(torch.randn(self.n_anchors, self.embedding_dim), requires_grad=True)

        self.user_embed = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embed = nn.Embedding(self.n_items, self.embedding_dim)

        self.user_embed.apply(general_weight_init)
        self.item_embed.apply(general_weight_init)

        self._acc_exc = 0
        self._acc_inc = 0

        self.name = 'ACF'

        print(f'Built {self.name} model \n'
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

        # Regularization losses
        _, c_i, c_i_unnorm = i_repr
        # Exclusiveness constraint
        exc_values = c_i * (c_i_unnorm - torch.logsumexp(c_i_unnorm, dim=-1, keepdim=True))
        exc = - exc_values.sum()

        # Inclusiveness constraint
        q_k = c_i.sum(dim=[0, 1]) / c_i.sum()  # n_anchors
        inc = - (q_k * torch.log(q_k)).sum()

        self._acc_exc += exc
        self._acc_inc += inc

        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_embed = self.user_embed(u_idxs)
        c_u = u_embed @ self.anchors.T  # [batch_size, n_anchors]
        c_u = nn.Softmax(dim=-1)(c_u)

        u_anc = c_u @ self.anchors  # [batch_size, embedding_dim]

        return u_anc

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_embed = self.item_embed(i_idxs)
        c_i_unnorm = i_embed @ self.anchors.T  # [batch_size, n_neg_p_1, embedding_dim]
        c_i = nn.Softmax(dim=-1)(c_i_unnorm)

        i_anc = c_i @ self.anchors
        return i_anc, c_i, c_i_unnorm

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        u_anc = u_repr
        i_anc = i_repr[0]
        dots = (u_anc.unsqueeze(-2) * i_anc).sum(dim=-1)
        return dots

    def get_and_reset_other_loss(self) -> float:
        _acc_exc, _acc_inc = self._acc_exc, self._acc_inc
        self._acc_exc = self._acc_inc = 0
        return self.delta_exc * _acc_exc - self.delta_inc * _acc_inc

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return ACF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_anchors'],
                   conf['delta_exc'], conf['delta_inc'])


class UProtoMF(SGDBasedRecommenderAlgorithm):
    """
    Implements the ProtoMF model with user prototypes
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

        self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]), requires_grad=True)

        self.cosine_sim_func = lambda x, y: (1 + nn.CosineSimilarity(dim=-1)(x, y)) / 2

        self.user_embed.apply(general_weight_init)
        self.item_embed.apply(general_weight_init)

        self._acc_r_proto = 0
        self._acc_r_batch = 0

        self.name = 'UProtoMF'

        print(f'Built {self.name} model \n'
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

        # Compute regularization losses
        sim_mtx = u_repr
        self._acc_r_proto += (1 - sim_mtx).min(dim=0).values.mean()
        self._acc_r_batch += (1 - sim_mtx).min(dim=1).values.mean()

        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_embed = self.user_embed(u_idxs)
        sim_mtx = self.cosine_sim_func(u_embed.unsqueeze(-2), self.prototypes)  # [batch_size,n_prototypes]
        return sim_mtx

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.item_embed(i_idxs)

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        dots = (u_repr.unsqueeze(-2) * i_repr).sum(dim=-1)
        return dots

    def get_and_reset_other_loss(self) -> float:
        acc_r_proto, acc_r_batch = self._acc_r_proto, self._acc_r_batch
        self._acc_r_proto = self._acc_r_batch = 0
        return self.sim_proto_weight * acc_r_proto + self.sim_batch_weight * acc_r_batch

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return UProtoMF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_prototypes'],
                        conf['sim_proto_weight'], conf['sim_batch_weight'])


class IProtoMF(SGDBasedRecommenderAlgorithm):
    """
    Implements the ProtoMF model with item prototypes
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

        self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]), requires_grad=True)

        self.cosine_sim_func = lambda x, y: (1 + nn.CosineSimilarity(dim=-1)(x, y)) / 2

        self.user_embed.apply(general_weight_init)
        self.item_embed.apply(general_weight_init)

        self._acc_r_proto = 0
        self._acc_r_batch = 0

        self.name = 'IProtoMF'

        print(f'Built {self.name} model \n'
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
        # Compute regularization losses
        sim_mtx = i_repr
        sim_mtx = sim_mtx.reshape(-1, self.n_prototypes)
        self._acc_r_proto += (1 - sim_mtx).min(dim=0).values.mean()
        self._acc_r_batch += (1 - sim_mtx).min(dim=1).values.mean()

        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.user_embed(u_idxs)

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_embed = self.item_embed(i_idxs)
        sim_mtx = self.cosine_sim_func(i_embed.unsqueeze(-2), self.prototypes)  # [batch_size,n_neg + 1,n_prototypes]

        return sim_mtx

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        dots = (u_repr.unsqueeze(-2) * i_repr).sum(dim=-1)
        return dots

    def get_and_reset_other_loss(self) -> float:
        acc_r_proto, acc_r_batch = self._acc_r_proto, self._acc_r_batch
        self._acc_r_proto = self._acc_r_batch = 0
        return self.sim_proto_weight * acc_r_proto + self.sim_batch_weight * acc_r_batch

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return IProtoMF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_prototypes'],
                        conf['sim_proto_weight'], conf['sim_batch_weight'])


class UIProtoMF(SGDBasedRecommenderAlgorithm):
    """
    Implements the ProtoMF model with item and user prototypes
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 100, u_n_prototypes: int = 20,
                 i_n_prototypes: int = 20, u_sim_proto_weight: float = 1., u_sim_batch_weight: float = 1.,
                 i_sim_proto_weight: float = 1., i_sim_batch_weight: float = 1.):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.u_n_prototypes = u_n_prototypes
        self.i_n_prototypes = i_n_prototypes
        self.u_sim_proto_weight = u_sim_proto_weight
        self.u_sim_batch_weight = u_sim_batch_weight
        self.i_sim_proto_weight = i_sim_proto_weight
        self.i_sim_batch_weight = i_sim_batch_weight

        self.user_embed = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embed = nn.Embedding(self.n_items, self.embedding_dim)

        self.u_prototypes = nn.Parameter(torch.randn([self.u_n_prototypes, self.embedding_dim]), requires_grad=True)
        self.i_prototypes = nn.Parameter(torch.randn([self.i_n_prototypes, self.embedding_dim]), requires_grad=True)

        self.u_to_i_proj = nn.Linear(self.embedding_dim, self.i_n_prototypes)
        self.i_to_u_proj = nn.Linear(self.embedding_dim, self.u_n_prototypes)

        self.cosine_sim_func = lambda x, y: (1 + nn.CosineSimilarity(dim=-1)(x, y)) / 2

        self.apply(general_weight_init)

        self._u_acc_r_proto = 0
        self._u_acc_r_batch = 0

        self._i_acc_r_proto = 0
        self._i_acc_r_batch = 0

        self.name = 'UIProtoMF'

        print(f'Built {self.name} model \n'
              f'- n_users: {self.n_users} \n'
              f'- n_items: {self.n_items} \n'
              f'- embedding_dim: {self.embedding_dim} \n'
              f'- u_n_prototypes: {self.u_n_prototypes} \n'
              f'- i_n_prototypes: {self.i_n_prototypes} \n'
              f'- u_sim_proto_weight: {self.u_sim_proto_weight} \n'
              f'- u_sim_batch_weight: {self.u_sim_batch_weight} \n'
              f'- i_sim_proto_weight: {self.i_sim_proto_weight} \n'
              f'- i_sim_batch_weight: {self.i_sim_batch_weight} \n'
              )

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_embed = self.user_embed(u_idxs)
        u_sim_mtx = self.cosine_sim_func(u_embed.unsqueeze(-2), self.u_prototypes)  # [batch_size, u_n_prototypes]
        u_proj = self.u_to_i_proj(u_embed)  # [batch_size, i_n_prototypes]

        return u_sim_mtx, u_proj

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_embed = self.item_embed(i_idxs)
        i_sim_mtx = self.cosine_sim_func(i_embed.unsqueeze(-2),
                                         self.i_prototypes)  # [batch_size, n_neg + 1, i_n_prototypes]
        i_proj = self.i_to_u_proj(i_embed)  # [batch_size, n_neg + 1, u_n_prototypes]

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

        # Regularization losses
        u_sim_mtx, _ = u_repr
        i_sim_mtx, _ = i_repr
        i_sim_mtx = i_sim_mtx.reshape(-1, self.i_n_prototypes)
        self._u_acc_r_proto += (1 - u_sim_mtx).min(dim=0).values.mean()
        self._u_acc_r_batch += (1 - u_sim_mtx).min(dim=1).values.mean()
        self._i_acc_r_proto += (1 - i_sim_mtx).min(dim=0).values.mean()
        self._i_acc_r_batch += (1 - i_sim_mtx).min(dim=1).values.mean()

        return dots

    def get_and_reset_other_loss(self) -> float:
        u_acc_r_proto, u_acc_r_batch = self._u_acc_r_proto, self._u_acc_r_batch
        i_acc_r_proto, i_acc_r_batch = self._i_acc_r_proto, self._i_acc_r_batch
        self._u_acc_r_proto = self._u_acc_r_batch = self._i_acc_r_proto = self._i_acc_r_batch = 0
        u_reg = self.u_sim_proto_weight * u_acc_r_proto + self.u_sim_batch_weight * u_acc_r_batch
        i_reg = self.i_sim_proto_weight * i_acc_r_proto + self.i_sim_batch_weight * i_acc_r_batch
        return u_reg + i_reg

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return UIProtoMF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['u_n_prototypes'],
                         conf['i_n_prototypes'], conf['u_sim_proto_weight'], conf['u_sim_batch_weight'],
                         conf['i_sim_proto_weight'], conf['i_sim_batch_weight'])
