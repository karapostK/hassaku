import logging
import os.path
from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict

import torch
from scipy import sparse as sp
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

        logging.info(f'Built {self.name} module')

    @abstractmethod
    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
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


class SparseMatrixBasedRecommenderAlgorithm(RecommenderAlgorithm, ABC):
    """
    Base class for Recommender System algorithms that can be trained using the sparse user-item matrix
    """

    def __init__(self):
        super().__init__()
        self.name = 'SparseMatrixBasedRecommenderAlgorithm'

        self.pred_mtx = None
        logging.info(f'Built {self.name} module')

    @abstractmethod
    def fit(self, matrix: sp.spmatrix):
        """
        :param matrix: user x item sparse matrix
        """
        pass

    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor):
        """
        Returns the score between users and items
        @param u_idxs: User indexes. Shape is [batch_size]
        @param i_idxs: Item indexes. Shape is [batch_size, n] where n is an arbitrary number of items. Usually all items.
        @return:
        """
        assert self.pred_mtx is not None, 'Prediction Matrix not computed, run fit!'
        if sp.issparse(self.pred_mtx):
            self.pred_mtx = self.pred_mtx.toarray()  # Not elegant but it simplifies the following code
        out = self.pred_mtx[u_idxs[:, None], i_idxs]
        return out


class SGDBasedRecommenderAlgorithm(RecommenderAlgorithm, ABC, nn.Module):
    """
    Base class for Recommender System algorithms based on iterative updates with stochastic gradient descent. It requires a Trainer object to update.
    """

    def __init__(self):
        super().__init__()
        self.name = 'SGDBasedRecommenderAlgorithm'

        logging.info(f'Built {self.name} module')

    def forward(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        """
        Similar to predict but used for training. It provides a simple default implementation that can be adjusted in
        case.
        """
        u_repr = self.get_user_representations(u_idxs)
        i_repr = self.get_item_representations(i_idxs)

        out = self.combine_user_item_representations(u_repr, i_repr)
        return out

    @abstractmethod
    def get_user_representations(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Returns a user representation given the user indexes. It is especially useful for faster validation.
        :param u_idxs: user indexes. Shape is (batch_size)
        :return: user representation/s. The output depends on the model.
        """
        pass

    @abstractmethod
    def get_item_representations(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Returns an item representation given the user indexes. It is especially useful for faster validation.
        :param i_idxs: item indexes. Shape is (batch_size, n_neg + 1), where n_neg is the number of negative samples
        :return: item representation/s. The output depends on the model.
        """
        pass

    @abstractmethod
    def combine_user_item_representations(self, u_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                                          i_repr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """
        Combine the user and item representations to generate the final logits.
        :param u_repr: User representations (see get_user_representations)
        :param i_repr: Item representations (see get_item_representations)
        :return:
        """
        pass

    def get_and_reset_other_loss(self) -> Dict:
        """
        Gets and reset the value of other losses defined for the specific module which go beyond the recommender system
        loss and the l2 loss. For example an entropy-based loss for ProtoMF. This function is always called by Trainer
        at the end of the batch pass to get the full loss! Be sure to implement it when the algorithm has additional losses!
        Note that the dictionary should contain AT LEAST the entry "reg_loss" which contains the aggregated other losses.
        "reg_loss" is used then for both updating the parameters and logging. The other values in the dictionary are just logged.
        :return: dictionary containing at least reg_loss
        """
        return {'reg_loss': torch.zeros(1)}

    @torch.no_grad()
    def predict(self, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        self.eval()
        out = self(u_idxs, i_idxs)
        return out

    def save_model_to_path(self, path: str):
        path = os.path.join(path, 'model.pth')
        torch.save(self.state_dict(), path)
        print('Model Saved')

    def load_model_from_path(self, path: str):
        path = os.path.join(path, 'model.pth')
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        print('Model Loaded')


class PrototypeWrapper(ABC, SGDBasedRecommenderAlgorithm):
    """
    Simple class to augment existing ones with methods related to Prototypes.
    """

    def get_item_representations_pre_tune(self, i_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        raise NotImplementedError('This method has not been implemented for this class!')

    def get_item_representations_post_tune(self, i_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        raise NotImplementedError('This method has not been implemented for this class!')

    def get_user_representations_pre_tune(self, u_idxs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        raise NotImplementedError('This method has not been implemented for this class!')

    def get_user_representations_post_tune(self, u_repr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        raise NotImplementedError('This method has not been implemented for this class!')
