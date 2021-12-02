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
        self.item_vectors = nn.Embedding.from_pretrained(self.user_vectors.weight.T) # todo: is it better to assign the weights later?

        # Initialization of the network
        self.user_nn.apply(general_weight_init)
        self.item_nn.apply(general_weight_init)

        self.name = 'DeepMatrixFactorization'

        print('DeepMatrixFactorization class created')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor, device: str = 'cpu'):

        # User pass
        u_vec = self.user_vectors(u_idxs)
        u_vec = self.user_nn(u_vec)

        # Item pass

        i_vec = self.item_vectors(i_idxs)
        i_vec = i_vec.to(device)
        i_vec = self.item_nn(i_vec)

        # Cosine
        sim = self.cosine_func(u_vec[:, None, :], i_vec)

        return sim

    @staticmethod
    def sparse_to_tensor(array):
        return torch.tensor(array.toarray(), dtype=torch.float)

    @torch.no_grad()
    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> typing.Union:
        self.eval()
        out = self(u_idxs, i_idxs)
        return out
