import typing
from abc import ABC, abstractmethod

import torch
from torch import nn


class RecommenderAlgorithm(ABC):
    """
    It implements the basic class for a Recommender System Algorithm.
    Each new algorithm has to override the method 'pred'.
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


class SGDBasedRecommenderAlgorithm(RecommenderAlgorithm, ABC, nn.Module):
    """

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
