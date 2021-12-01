import typing

import torch
from scipy import sparse as sp
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from base_classes import RecommenderAlgorithm
from utilities.utils import general_weight_init


class DeepMatrixFactorization(RecommenderAlgorithm):
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

        self.matrix = matrix
        self.n_users, self.n_items = self.matrix.shape

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

        self.apply(general_weight_init)

        self.name = 'DeepMatrixFactorization'

        print('DeepMatrixFactorization class created')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor, device: str = 'cpu'):

        # User pass
        u_vec = self.sparse_to_tensor(self.matrix[u_idxs])
        u_vec = u_vec.to(device)
        u_vec = self.user_nn(u_vec)

        # Item pass

        i_vec = self.sparse_to_tensor(self.matrix[:, i_idxs.flatten()]).T
        i_vec = i_vec.to(device)
        i_vec = self.item_nn(i_vec)
        i_vec = i_vec.reshape(list(i_idxs.shape) + [-1])

        # Cosine
        sim = self.cosine_func(u_vec[:, None, :], i_vec)

        return sim

    @staticmethod
    def sparse_to_tensor(array):
        return torch.tensor(array.toarray(), dtype=torch.float)

    def fit(self, train_loader: DataLoader, n_epochs: int = 50, lr: float = 1e-4, device: str = 'cuda'):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)

        for epoch in trange(n_epochs):

            self.train()

            epoch_train_loss = 0

            for u_idxs, i_idxs, labels in tqdm(train_loader):
                u_idxs = u_idxs
                i_idxs = i_idxs
                labels = labels.to(device)

                out = self(u_idxs, i_idxs, device)

                loss = nn.BCEWithLogitsLoss()(out.flatten(), labels.flatten())

                epoch_train_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_loss /= len(train_loader)
            print("Epoch {} - Epoch Avg Train Loss {:.3f} \n".format(epoch, epoch_train_loss))

    @torch.no_grad()
    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> typing.Union:
        self.eval()
        out = self(u_idxs, i_idxs)
        return out


class dmf(nn.Module):
    def __init__(self, matrix: sp.spmatrix, middle_dim: int = 100, final_dim: int = 64, device='cpu'):
        super(dmf, self).__init__()
        self.matrix = matrix
        self.n_users = matrix.shape[0]
        self.n_items = matrix.shape[1]
        self.middle_dim = middle_dim
        self.final_dim = final_dim
        self.device = device

        # Going with a two-layer NN
        self.user_nn = nn.Sequential(
            nn.Linear(self.n_items, self.middle_dim),
            nn.ReLU(),
            nn.Linear(self.middle_dim, self.final_dim)
        )

        self.item_nn = nn.Sequential(
            nn.Linear(self.n_users, self.middle_dim),
            nn.ReLU(),
            nn.Linear(self.middle_dim, self.final_dim)
        )
        self.cosine_fun = nn.CosineSimilarity(dim=-1)

        self.apply(general_weight_init)
