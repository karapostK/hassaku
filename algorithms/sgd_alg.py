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

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        out = self.user_bias(u_idxs) + self.item_bias(i_idxs).squeeze() + self.global_bias

        return out

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        return SGDBaseline(dataset.n_users, dataset.n_items)


class SGDMatrixFactorization(SGDBasedRecommenderAlgorithm):
    """
    Implements a simple Matrix Factorization model trained with gradient descent
    """

    def __init__(self, n_users: int, n_items: int, latent_dimension: int = 100, use_user_bias: bool = False,
                 use_item_bias: bool = False, use_global_bias: bool = False):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.latent_dimension = latent_dimension

        self.use_user_bias = use_user_bias
        self.use_item_bias = use_item_bias
        self.use_global_bias = use_global_bias

        self.user_embeddings = nn.Embedding(self.n_users, self.latent_dimension)
        self.item_embeddings = nn.Embedding(self.n_items, self.latent_dimension)

        if self.use_user_bias:
            self.user_bias = nn.Embedding(self.n_users, 1)
        if self.use_item_bias:
            self.item_bias = nn.Embedding(self.n_items, 1)

        self.apply(general_weight_init)

        if self.use_global_bias:
            self.global_bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.name = 'SGDMatrixFactorization'

        print(f'Built {self.name} module\n'
              f'- latent_dimension: {self.latent_dimension} \n'
              f'- use_user_bias: {self.use_user_bias} \n'
              f'- use_item_bias: {self.use_item_bias} \n'
              f'- use_global_bias: {self.use_global_bias}')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_embed = self.user_embeddings(u_idxs)
        i_embed = self.item_embeddings(i_idxs)

        out = (u_embed[:, None, :] * i_embed).sum(axis=-1)

        if self.use_user_bias:
            u_bias = self.user_bias(u_idxs)
            out += u_bias
        if self.use_item_bias:
            i_bias = self.item_bias(i_idxs).squeeze()
            out += i_bias
        if self.use_global_bias:
            out += self.global_bias

        return out

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        return SGDMatrixFactorization(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['use_user_bias'],
                                      conf['use_item_bias'], conf['use_global_bias'])


class UProtoMF(SGDBasedRecommenderAlgorithm):
    """
    Implements the ProtoMF model with user prototypes
    """

    def __init__(self, n_users: int, n_items: int, latent_dimension: int = 100, n_prototypes: int = 20,
                 sim_proto_weight: float = 1., sim_batch_weight: float = 1.):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.latent_dimension = latent_dimension
        self.n_prototypes = n_prototypes
        self.sim_proto_weight = sim_proto_weight
        self.sim_batch_weight = sim_batch_weight

        self.user_embed = nn.Embedding(self.n_users, self.latent_dimension)
        self.item_embed = nn.Embedding(self.n_items, self.n_prototypes)

        self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.latent_dimension]))

        self.cosine_sim_func = lambda x, y: (1 + nn.CosineSimilarity(dim=-1)(x, y)) / 2

        self.user_embed.apply(general_weight_init)
        self.item_embed.apply(general_weight_init)

        self._acc_r_proto = 0
        self._acc_r_batch = 0

        self.name = 'UProtoMF'

        print(f'Built {self.name} model \n'
              f'- n_users: {self.n_users} \n'
              f'- n_items: {self.n_items} \n'
              f'- latent_dimension: {self.latent_dimension} \n'
              f'- n_prototypes: {self.n_prototypes} \n'
              f'- sim_proto_weight: {self.sim_proto_weight} \n'
              f'- sim_batch_weight: {self.sim_batch_weight} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_embed = self.user_embed(u_idxs)
        i_embed = self.item_embed(i_idxs)

        sim_mtx = self.cosine_sim_func(u_embed.unsqueeze(-2), self.prototypes)  # [batch_size,n_prototypes]

        dots = (sim_mtx.unsqueeze(-2) * i_embed).sum(axis=-1)

        # Compute regularization losses
        self._acc_r_proto += - sim_mtx.max(dim=0).values.mean()
        self._acc_r_batch += - sim_mtx.max(dim=1).values.mean()

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

    def __init__(self, n_users: int, n_items: int, latent_dimension: int = 100, n_prototypes: int = 20,
                 sim_proto_weight: float = 1., sim_batch_weight: float = 1.):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.latent_dimension = latent_dimension
        self.n_prototypes = n_prototypes
        self.sim_proto_weight = sim_proto_weight
        self.sim_batch_weight = sim_batch_weight

        self.user_embed = nn.Embedding(self.n_users, self.n_prototypes)
        self.item_embed = nn.Embedding(self.n_items, self.latent_dimension)

        self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.latent_dimension]))

        self.cosine_sim_func = lambda x, y: (1 + nn.CosineSimilarity(dim=-1)(x, y)) / 2

        self.user_embed.apply(general_weight_init)
        self.item_embed.apply(general_weight_init)

        self._acc_r_proto = 0
        self._acc_r_batch = 0

        self.name = 'IProtoMF'

        print(f'Built {self.name} model \n'
              f'- n_users: {self.n_users} \n'
              f'- n_items: {self.n_items} \n'
              f'- latent_dimension: {self.latent_dimension} \n'
              f'- n_prototypes: {self.n_prototypes} \n'
              f'- sim_proto_weight: {self.sim_proto_weight} \n'
              f'- sim_batch_weight: {self.sim_batch_weight} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_embed = self.user_embed(u_idxs)
        i_embed = self.item_embed(i_idxs)

        sim_mtx = self.cosine_sim_func(i_embed.unsqueeze(-2), self.prototypes)  # [batch_size,n_neg + 1,n_prototypes]

        dots = (u_embed.unsqueeze(-2) * sim_mtx).sum(axis=-1)

        # Compute regularization losses
        sim_mtx = sim_mtx.reshape(-1, self.n_prototypes)
        self._acc_r_proto += - sim_mtx.max(dim=0).values.mean()
        self._acc_r_batch += - sim_mtx.max(dim=1).values.mean()

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

    def __init__(self, n_users: int, n_items: int, latent_dimension: int = 100, u_n_prototypes: int = 20,
                 i_n_prototypes: int = 20, u_sim_proto_weight: float = 1., u_sim_batch_weight: float = 1.,
                 i_sim_proto_weight: float = 1., i_sim_batch_weight: float = 1.):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.latent_dimension = latent_dimension
        self.u_n_prototypes = u_n_prototypes
        self.i_n_prototypes = i_n_prototypes
        self.u_sim_proto_weight = u_sim_proto_weight
        self.u_sim_batch_weight = u_sim_batch_weight
        self.i_sim_proto_weight = i_sim_proto_weight
        self.i_sim_batch_weight = i_sim_batch_weight

        self.user_embed = nn.Embedding(self.n_users, self.latent_dimension)
        self.item_embed = nn.Embedding(self.n_items, self.latent_dimension)

        self.u_prototypes = nn.Parameter(torch.randn([self.u_n_prototypes, self.latent_dimension]))
        self.i_prototypes = nn.Parameter(torch.randn([self.i_n_prototypes, self.latent_dimension]))

        self.u_to_i_proj = nn.Linear(self.latent_dimension, self.i_n_prototypes)
        self.i_to_u_proj = nn.Linear(self.latent_dimension, self.u_n_prototypes)

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
              f'- latent_dimension: {self.latent_dimension} \n'
              f'- u_n_prototypes: {self.u_n_prototypes} \n'
              f'- i_n_prototypes: {self.i_n_prototypes} \n'
              f'- u_sim_proto_weight: {self.u_sim_proto_weight} \n'
              f'- u_sim_batch_weight: {self.u_sim_batch_weight} \n'
              f'- i_sim_proto_weight: {self.i_sim_proto_weight} \n'
              f'- i_sim_batch_weight: {self.i_sim_batch_weight} \n'
              )

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_embed = self.user_embed(u_idxs)  # [batch_size, latent_dimension]
        i_embed = self.item_embed(i_idxs)  # [batch_size, n_neg + 1, latent_dimension]

        # User pass
        u_sim_mtx = self.cosine_sim_func(u_embed.unsqueeze(-2), self.u_prototypes)  # [batch_size, u_n_prototypes]
        i_proj = self.i_to_u_proj(i_embed)  # [batch_size, n_neg + 1, u_n_prototypes]

        u_dots = (u_sim_mtx.unsqueeze(-2) * i_proj).sum(axis=-1)

        self._u_acc_r_proto += - u_sim_mtx.max(dim=0).values.mean()
        self._u_acc_r_batch += - u_sim_mtx.max(dim=1).values.mean()

        # Item pass
        i_sim_mtx = self.cosine_sim_func(i_embed.unsqueeze(-2),
                                         self.i_prototypes)  # [batch_size, n_neg + 1, i_n_prototypes]
        u_proj = self.u_to_i_proj(u_embed)  # [batch_size, i_n_prototypes]

        i_dots = (u_proj.unsqueeze(-2) * i_sim_mtx).sum(axis=-1)

        i_sim_mtx = i_sim_mtx.reshape(-1, self.i_n_prototypes)
        self._i_acc_r_proto += - i_sim_mtx.max(dim=0).values.mean()
        self._i_acc_r_batch += - i_sim_mtx.max(dim=1).values.mean()

        # Merge
        dots = u_dots + i_dots
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
