import typing

import numpy as np
import torch

from algorithms.base_classes import RecommenderAlgorithm


class RandomItems(RecommenderAlgorithm):

    def __init__(self):
        super().__init__()
        self.name = 'RandomItems'
        print(f'Built {self.name} module')

    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> typing.Union[np.ndarray, torch.Tensor]:
        # Generate random scores
        out = torch.rand(i_idxs.shape)
        return out


class PopularItems(RecommenderAlgorithm):

    def __init__(self, pop_distribution: np.ndarray):
        """
        Assign a score to each item depending on its popularity.
        :param pop_distribution: array with shape (n_items,). For each item we have its popularity value.
        """
        super().__init__()
        self.pop_distribution = pop_distribution
        self.name = 'PopularItems'
        print(f'Built {self.name} module')

    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> typing.Union[np.ndarray, torch.Tensor]:
        out = self.pop_distribution[i_idxs]
        return out
