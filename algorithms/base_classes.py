import typing
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange


class RecommenderAlgorithm(nn.Module, ABC):
    """
    It implements the basic class for a Recommender System Algorithm.
    Each new algorithm has to override the method 'pred'.
    """

    def __init__(self):
        super().__init__()
        self.name = 'RecommenderAlgorithm'

        print('RecommenderAlgorithm class crated')

    @abstractmethod
    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> typing.Union:
        """
        Predict the affinity score for the users in u_idxs over the items i_idxs.
        NB. The methods have to take into account of the negative sampling! (see the shape of the inputs)
        :param u_idxs: user indexes. Shape is (batch_size)
        :param i_idxs: item indexes. Shape is (batch_size, n_neg + 1), where n_neg is the number of negative samples
        and 1 is the positive item.
        :return preds: predictions. Shape is (batch_size, n_neg + 1)
        """


class SGDBasedRecommenderAlgorithm(RecommenderAlgorithm, ABC):
    """
    It extends the previous class by providing a fit function that perform a standard training procedure
    """

    def __init__(self):
        super().__init__()
        self.name = 'SGDBasedRecommenderAlgorithm'

        print('SGDBasedRecommenderAlgorithm class crated')

    # todo incorporate the parameters from Protorec e.g. changing the loss function/optimizer
    def fit(self, train_loader: DataLoader, n_epochs: int = 50, lr: float = 1e-4, device: str = 'cuda', **kwargs):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)

        for epoch in trange(n_epochs):

            self.train()

            epoch_train_loss = 0

            for u_idxs, i_idxs, labels in tqdm(train_loader):
                u_idxs = u_idxs.to(device)
                i_idxs = i_idxs.to(device)
                labels = labels.to(device)

                out = self(u_idxs, i_idxs, device)

                loss = nn.BCEWithLogitsLoss()(out.flatten(), labels.flatten())

                epoch_train_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_loss /= len(train_loader)
            print("Epoch {} - Epoch Avg Train Loss {:.3f} \n".format(epoch, epoch_train_loss))
