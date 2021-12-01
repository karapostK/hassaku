import typing
from abc import ABC, abstractmethod

import torch
from torch import nn


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
