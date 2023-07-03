import logging
import warnings
from io import BytesIO
from typing import Union, Tuple, Dict

import matplotlib
import torch
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import nn
from torch.nn import functional as F
from torch.utils import data

from algorithms.base_classes import SGDBasedRecommenderAlgorithm
from train.utils import general_weight_init

warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('agg')


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
        c_i_unnorm = i_embed @ self.anchors.T  # [batch_size, n_neg_p_1, n_anchors]
        c_i = nn.Softmax(dim=-1)(c_i_unnorm)

        i_anc = c_i @ self.anchors
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
            'reg_loss': exc_loss - inc_loss,
            'exc_loss': exc_loss,
            'inc_loss': inc_loss
        }

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return ACF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['n_anchors'],
                   conf['delta_exc'], conf['delta_inc'])


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

        # Compute regularization losses
        sim_mtx = u_repr
        dis_mtx = (1 - sim_mtx)
        self._acc_r_proto += dis_mtx.min(dim=0).values.mean()
        self._acc_r_batch += dis_mtx.min(dim=1).values.mean()

        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_embed = self.user_embed(u_idxs)
        sim_mtx = compute_norm_cosine_sim(u_embed, self.prototypes)  # [batch_size, n_prototypes]

        return sim_mtx

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.item_embed(i_idxs)

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        dots = (u_repr.unsqueeze(-2) * i_repr).sum(dim=-1)
        return dots

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
        # This function is run at the end of the evaluation procedure.
        with torch.no_grad():
            proto_users = torch.cat([self.prototypes, self.user_embed.weight])
            sim_mtx = compute_norm_cosine_sim(proto_users, proto_users)
            dis_mtx = 1 - sim_mtx

            # Average pairwise distance between prototypes
            dis_mtx_proto = dis_mtx[:self.n_prototypes, :self.n_prototypes]
            dis_mtx_proto_tril = torch.tril(dis_mtx_proto, diagonal=-1)
            avg_pairwise_proto_dis = (dis_mtx_proto_tril.sum() * 2) / (self.n_prototypes * (self.n_prototypes - 1))

            # User-to-prototype relatedness

            user_to_proto = sim_mtx[self.n_prototypes:, :self.n_prototypes]
            user_to_proto_mean = user_to_proto.mean(dim=-1).mean()
            user_to_proto_mean_adjusted = user_to_proto_mean * self.n_prototypes
            user_to_proto_max = user_to_proto.max(dim=-1).values.mean()
            user_to_proto_min = user_to_proto.min(dim=-1).values.mean()

            # Entropy
            user_to_proto_soft = nn.Softmax(dim=-1)(user_to_proto)
            user_to_proto_entropy = - user_to_proto_soft * (
                    user_to_proto - torch.logsumexp(user_to_proto, dim=-1, keepdim=True))
            user_to_proto_entropy = user_to_proto_entropy.mean()

            dis_mtx = dis_mtx.cpu()
            dis_mtx_proto = dis_mtx_proto.cpu()
            avg_pairwise_proto_dis = avg_pairwise_proto_dis.item()
            user_to_proto_mean = user_to_proto_mean.item()
            user_to_proto_max = user_to_proto_max.item()
            user_to_proto_min = user_to_proto_min.item()
            user_to_proto_mean_adjusted = user_to_proto_mean_adjusted.item()
            user_to_proto_entropy = user_to_proto_entropy.item()

        # Compute TSNE
        tsne = TSNE(learning_rate='auto', metric='precomputed')
        tsne_results = tsne.fit_transform(dis_mtx)
        proto_tsne = tsne_results[:self.n_prototypes]
        user_tsne = tsne_results[self.n_prototypes:]
        plt.figure(figsize=(6, 6), dpi=200)
        plt.scatter(user_tsne[:, 0], user_tsne[:, 1], s=10, alpha=0.6, c='#74add1', label='Users')
        plt.scatter(proto_tsne[:, 0], proto_tsne[:, 1], s=30, c='#d73027', alpha=0.9, label='Prototypes')
        plt.axis('off')
        plt.tight_layout()
        plt.legend(loc="upper left", prop={'size': 13})

        # Saving it in memory and Loading as PIL
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        latent_space_image = wandb.Image(Image.open(buffer), caption="Epoch: {}".format(curr_epoch))

        # Computing distance matrix for prototypes
        plt.figure(figsize=(5, 5), dpi=100)
        plt.matshow(dis_mtx_proto, vmin=0, vmax=1, fignum=1, cmap='cividis')

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        dis_mtx_proto_image = wandb.Image(Image.open(buffer), caption="Epoch: {}".format(curr_epoch))

        return {'latent_space': latent_space_image,
                'dis_mtx_proto': dis_mtx_proto_image,
                'avg_pairwise_proto_dis': avg_pairwise_proto_dis,
                'user_to_proto_mean': user_to_proto_mean,
                'user_to_proto_max': user_to_proto_max,
                'user_to_proto_min': user_to_proto_min,
                'user_to_proto_mean_adjusted': user_to_proto_mean_adjusted,
                'user_to_proto_entropy': user_to_proto_entropy
                }


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
        # Compute regularization losses
        sim_mtx = i_repr
        sim_mtx = sim_mtx.reshape(-1, self.n_prototypes)
        dis_mtx = (1 - sim_mtx)
        self._acc_r_proto += dis_mtx.min(dim=0).values.mean()
        self._acc_r_batch += dis_mtx.min(dim=1).values.mean()

        return dots

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.user_embed(u_idxs)

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_embed = self.item_embed(i_idxs)
        i_embed = i_embed.reshape(-1, i_embed.shape[-1])
        sim_mtx = compute_norm_cosine_sim(i_embed, self.prototypes)
        sim_mtx = sim_mtx.reshape(list(i_idxs.shape) + [self.n_prototypes])

        return sim_mtx

    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        dots = (u_repr.unsqueeze(-2) * i_repr).sum(dim=-1)
        return dots

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
        # This function is run at the end of the evaluation procedure.
        with torch.no_grad():
            item_sample = self.item_embed.weight
            if item_sample.shape[0] >= 10000:
                indxs = torch.randperm(item_sample.shape[0])[:10000]
                item_sample = item_sample[indxs]

            proto_items = torch.cat([self.prototypes, item_sample])
            sim_mtx = compute_norm_cosine_sim(proto_items, proto_items)
            dis_mtx = 1 - sim_mtx

            # Average pairwise distance between prototypes
            dis_mtx_proto = dis_mtx[:self.n_prototypes, :self.n_prototypes]
            dis_mtx_proto_tril = torch.tril(dis_mtx_proto, diagonal=-1)
            avg_pairwise_proto_dis = (dis_mtx_proto_tril.sum() * 2) / (self.n_prototypes * (self.n_prototypes - 1))

            # Item-to-prototype relatedness

            item_to_proto = sim_mtx[self.n_prototypes:, :self.n_prototypes]
            item_to_proto_mean = item_to_proto.mean(dim=-1).mean()
            item_to_proto_mean_adjusted = item_to_proto_mean * self.n_prototypes
            item_to_proto_max = item_to_proto.max(dim=-1).values.mean()
            item_to_proto_min = item_to_proto.min(dim=-1).values.mean()

            # Entropy
            item_to_proto_soft = nn.Softmax(dim=-1)(item_to_proto)
            item_to_proto_entropy = - item_to_proto_soft * (
                    item_to_proto - torch.logsumexp(item_to_proto, dim=-1, keepdim=True))
            item_to_proto_entropy = item_to_proto_entropy.mean()

            dis_mtx = dis_mtx.cpu()
            dis_mtx_proto = dis_mtx_proto.cpu()
            avg_pairwise_proto_dis = avg_pairwise_proto_dis.item()
            item_to_proto_mean = item_to_proto_mean.item()
            item_to_proto_max = item_to_proto_max.item()
            item_to_proto_min = item_to_proto_min.item()
            item_to_proto_mean_adjusted = item_to_proto_mean_adjusted.item()
            item_to_proto_entropy = item_to_proto_entropy.item()

        tsne = TSNE(learning_rate='auto', metric='precomputed')
        tsne_results = tsne.fit_transform(dis_mtx)
        proto_tsne = tsne_results[:self.n_prototypes]
        item_tsne = tsne_results[self.n_prototypes:]
        plt.figure(figsize=(6, 6), dpi=200)
        plt.scatter(item_tsne[:, 0], item_tsne[:, 1], s=10, alpha=0.6, c='#74add1', label='Items')
        plt.scatter(proto_tsne[:, 0], proto_tsne[:, 1], s=30, c='#d73027', alpha=0.9, label='Prototypes')
        plt.axis('off')
        plt.tight_layout()
        plt.legend(loc="upper left", prop={'size': 13})

        # Saving it in memory and Loading as PIL
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        latent_space_image = wandb.Image(Image.open(buffer), caption="Epoch: {}".format(curr_epoch))

        # Computing distance matrix for prototypes
        plt.figure(figsize=(5, 5), dpi=100)
        plt.matshow(dis_mtx_proto, vmin=0, vmax=1, fignum=1, cmap='cividis')

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        dis_mtx_proto_image = wandb.Image(Image.open(buffer), caption="Epoch: {}".format(curr_epoch))

        return {'latent_space': latent_space_image,
                'dis_mtx_proto': dis_mtx_proto_image,
                'avg_pairwise_proto_dis': avg_pairwise_proto_dis,
                'item_to_proto_mean': item_to_proto_mean,
                'item_to_proto_max': item_to_proto_max,
                'item_to_proto_min': item_to_proto_min,
                'item_to_proto_mean_adjusted': item_to_proto_mean_adjusted,
                'item_to_proto_entropy': item_to_proto_entropy
                }


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

        self.uprotomf = UProtoMF(n_users, n_items, embedding_dim, u_n_prototypes,
                                 u_sim_proto_weight, u_sim_batch_weight)

        self.iprotomf = IProtoMF(n_users, n_items, embedding_dim, i_n_prototypes,
                                 i_sim_proto_weight, i_sim_batch_weight)

        self.u_to_i_proj = nn.Linear(self.embedding_dim, i_n_prototypes, bias=False)
        self.i_to_u_proj = nn.Linear(self.embedding_dim, u_n_prototypes, bias=False)

        self.uprotomf.get_item_representations = nn.Sequential(
            self.iprotomf.item_embed,
            self.i_to_u_proj
        )

        self.iprotomf.get_user_representations = nn.Sequential(
            self.uprotomf.user_embed,
            self.u_to_i_proj
        )

        self.apply(general_weight_init)

        self._u_acc_r_proto = 0
        self._u_acc_r_batch = 0

        self._i_acc_r_proto = 0
        self._i_acc_r_batch = 0

        self.name = 'UIProtoMF'

        logging.info(f'Built {self.name} model \n')

    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        u_sim_mtx = self.uprotomf.get_user_representations(u_idxs)
        u_proj = self.iprotomf.get_user_representations(u_idxs)

        return u_sim_mtx, u_proj

    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        i_sim_mtx = self.iprotomf.get_item_representations(i_idxs)
        i_proj = self.uprotomf.get_item_representations(i_idxs)

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
        u_dots = self.uprotomf.forward(u_idxs, i_idxs)
        i_dots = self.iprotomf.forward(u_idxs, i_idxs)

        dots = u_dots + i_dots

        return dots

    def get_and_reset_other_loss(self) -> Dict:
        u_reg = {'user_' + k: v for k, v in self.uprotomf.get_and_reset_other_loss().items()}
        i_reg = {'item_' + k: v for k, v in self.iprotomf.get_and_reset_other_loss().items()}
        return {
            'reg_loss': u_reg.pop('user_reg_loss') + i_reg.pop('item_reg_loss'),
            **u_reg,
            **i_reg
        }

    @staticmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        return UIProtoMF(dataset.n_users, dataset.n_items, conf['embedding_dim'], conf['u_n_prototypes'],
                         conf['i_n_prototypes'], conf['u_sim_proto_weight'], conf['u_sim_batch_weight'],
                         conf['i_sim_proto_weight'], conf['i_sim_batch_weight'])
