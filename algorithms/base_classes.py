import typing
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.utils import data


class RecommenderAlgorithm(ABC):
    """
    It implements the basic class for a Recommender System Algorithm.
    Each new algorithm has to override the method 'predict'.
    """

    def __init__(self):
        super().__init__()
        self.name = 'RecommenderAlgorithm'

        print(f'Built {self.name} module')

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

    @abstractmethod
    def save_model_to_path(self, path: str):
        """
        Saves the necessary data to reconstruct this class to a specified path
        """

    @abstractmethod
    def load_model_from_path(self, path: str):
        """
        Load the necessary data to reconstruct a previous model from a specified path
        """

    @staticmethod
    @abstractmethod
    def build_from_conf(conf: dict, dataset: data.Dataset):
        """
        Build the class from a configuration dictionary
        """


class SGDBasedRecommenderAlgorithm(RecommenderAlgorithm, ABC, nn.Module):
    """
    Base class for Recommender System algorithms based on iterative updates with stochastic gradient descent. It requires a Trainer object to update.
    """

    def __init__(self):
        super().__init__()
        self.name = 'SGDBasedRecommenderAlgorithm'

        print(f'Built {self.name} module')

    @abstractmethod
    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        """
        Similar to predict but used mostly for training
        """

    def get_and_reset_other_loss(self) -> float:
        """
        Gets and reset the value of other losses defined for the specific module which go beyond the recommender system
        loss and the l2 loss. For example an entropy-based loss for ProtoMF. This function is always called by Trainer
        at the end of the batch pass to get the full loss! Be sure to implement it when the algorithm has additional losses!
        :return: loss of the feature extractor
        """
        return 0

    @torch.no_grad()
    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> typing.Union:
        self.eval()
        out = self(u_idxs, i_idxs)
        return out

    def save_model_to_path(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model_from_path(self, path: str):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
