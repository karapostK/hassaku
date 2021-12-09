import typing

import torch
from scipy import sparse as sp
from torch import nn

from base_classes import SGDBasedRecommenderAlgorithm
from utilities.utils import general_weight_init


class DeepMatrixFactorization(SGDBasedRecommenderAlgorithm):
    """
    Deep Matrix Factorization Models for Recommender Systems by Xue et al. (https://www.ijcai.org/Proceedings/2017/0447.pdf)
    """

    def __init__(self, matrix: sp.spmatrix, u_mid_layers: typing.List[int], i_mid_layers: typing.List[int],
                 final_dimension: int):
        """
        :param matrix: user x item sparse matrix
        :param u_mid_layers: list of integers representing the size of the middle layers on the user side
        :param i_mid_layers: list of integers representing the size of the middle layers on the item side
        :param final_dimension: last dimension of the layers for both user and item side
        """
        super().__init__()

        self.n_users, self.n_items = matrix.shape

        self.final_dimension = final_dimension

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

        self.cosine_func = nn.CosineSimilarity(dim=-1)

        # Unfortunately, it seems that there is no CUDA support for sparse matrices..
        self.user_vectors = nn.Embedding.from_pretrained(torch.Tensor(matrix.todense()))
        self.item_vectors = nn.Embedding.from_pretrained(self.user_vectors.weight.T)

        # Initialization of the network
        self.user_nn.apply(general_weight_init)
        self.item_nn.apply(general_weight_init)

        self.name = 'DeepMatrixFactorization'

        print(f'Built {self.name} module \n'
              f'- u_layers: {self.u_layers} \n'
              f'- i_layers: {self.i_layers} \n')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor):

        # User pass
        u_vec = self.user_vectors(u_idxs)
        u_vec = self.user_nn(u_vec)

        # Item pass

        i_vec = self.item_vectors(i_idxs)
        i_vec = self.item_nn(i_vec)

        # Cosine
        sim = self.cosine_func(u_vec[:, None, :], i_vec)

        return sim


class SGDMatrixFactorization(SGDBasedRecommenderAlgorithm):
    """
    Implements a simple Matrix Factorization model trained with gradient descent
    It is similar to Probablistic Matrix Factorization (https://proceedings.neurips.cc/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf)
    """

    def __init__(self, n_users: int, n_items: int, latent_dimension: int = 100):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.latent_dimension = latent_dimension

        self.user_embeddings = nn.Embedding(self.n_users, self.latent_dimension)
        self.item_embeddings = nn.Embedding(self.n_items, self.latent_dimension)

        self.apply(general_weight_init)

        self.name = 'SGDMatrixFactorization'

        print(f'Built {self.name} module\n'
              f'- latent_dimension: {self.latent_dimension}')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_embed = self.user_embeddings(u_idxs)
        i_embed = self.item_embeddings(i_idxs)

        out = (u_embed[:, None, :] * i_embed).sum(axis=-1)

        return out


class GeneralizedMatrixFactorizationNCF(SGDMatrixFactorization):
    """
    One of the models proposed in Neural Collaborative Filtering (https://dl.acm.org/doi/pdf/10.1145/3038912.3052569)
    """

    def __init__(self, n_users: int, n_items: int, latent_dimension: int = 100, apply_last_layer: bool = True):
        """
        :param apply_last_layer: whether the last layer of the network should be applied.
        """
        super().__init__(n_users, n_items, latent_dimension)

        self.lin_projection = nn.Linear(latent_dimension, 1)
        self.apply_last_layer = apply_last_layer

        self.lin_projection.apply(general_weight_init)

        self.name = 'GeneralizedMatrixFactorization'

        print(f'Built {self.name} module \n'
              f'- apply_last_layer: {self.apply_last_layer}')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_embed = self.user_embeddings(u_idxs)
        i_embed = self.item_embeddings(i_idxs)

        out = u_embed[:, None, :] * i_embed
        if self.apply_last_layer:
            out = self.lin_projection(out).squeeze()

        return out


class MultiLayerPerceptronNCF(SGDMatrixFactorization):
    """
    One of the models proposed in Neural Collaborative Filtering (https://dl.acm.org/doi/pdf/10.1145/3038912.3052569)
    """

    def __init__(self, n_users: int, n_items: int, middle_layers: typing.List[int], latent_dimension: int = 100,
                 apply_last_layer: bool = True):
        """
        :param middle_layers: list of sizes of the middle layers. The initial layer size(2*latent_dimension) and the last layer (2) are automatically added
        :param apply_last_layer: whether the last layer of the network should be applied.
        """
        super().__init__(n_users, n_items, latent_dimension)

        self.layers = [self.latent_dimension * 2] + middle_layers + [1]
        self.apply_last_layer = apply_last_layer

        mlp = []
        for i, (n_in, n_out) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp.append(nn.Linear(n_in, n_out))

            if i != len(self.layers) - 2:
                mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

        self.mlp.apply(general_weight_init)

        self.name = 'MultiLayerPerceptronNCF'

        print(f'Built {self.name} module \n'
              f'- layers: {self.layers}'
              f'- apply_last_layer: {self.apply_last_layer}')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        u_embed = self.user_embeddings(u_idxs)
        i_embed = self.item_embeddings(i_idxs)

        u_embed = torch.repeat_interleave(u_embed[:, None, :], i_embed.shape[1], dim=1)

        ui_embed = torch.cat([u_embed, i_embed], dim=-1)

        if self.apply_last_layer:
            out = self.mlp(ui_embed).squeeze()
        else:
            out = self.mlp[:-1](ui_embed)

        return out


class NeuralMatrixFactorizationNCF(SGDBasedRecommenderAlgorithm):
    """
    From the paper (https://dl.acm.org/doi/pdf/10.1145/3038912.3052569). It combines MultiLayerPerceptronNCF and GeneralizedMatrixFactorizationNCF.
    """

    def __init__(self, n_users: int, n_items: int, middle_layers: typing.List[int], latent_dimension: int = 100):
        super().__init__()

        self.gmf = GeneralizedMatrixFactorizationNCF(n_users, n_items, latent_dimension, apply_last_layer=False)
        self.mlp = MultiLayerPerceptronNCF(n_users, n_items, middle_layers, latent_dimension, apply_last_layer=False)

        self.linear_projection = nn.Linear(latent_dimension + middle_layers[-1], 1)

        self.linear_projection.apply(general_weight_init)

        self.name = 'NeuralMatrixFactorizationNCF'

        print(f'Built {self.name} module')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        gmf_out = self.gmf(u_idxs, i_idxs)
        mlp_out = self.mlp(u_idxs, i_idxs)

        concatenated = torch.cat((gmf_out, mlp_out), dim=-1)

        out = self.linear_projection(concatenated).squeeze()

        return out
