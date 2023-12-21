import logging
import math
from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import nn

from eval.metrics import weight_ndcg_at_k_batch, weight_precision_at_k_batch
from utilities.modules import NeuralSort


class RecommenderSystemLoss(ABC):
    def __init__(self):
        super().__init__()
        self.name = 'RecommenderSystemLoss'

        logging.info(f'Built {self.name} module')

    @abstractmethod
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        pass

    @staticmethod
    @abstractmethod
    def build_from_conf(conf: dict, dataset):
        pass


class RecBinaryCrossEntropy(RecommenderSystemLoss):

    def __init__(self):
        super().__init__()
        self.name = 'RecBinaryCrossEntropy'

        logging.info(f'Built {self.name} module')

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        return RecBinaryCrossEntropy()

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        It computes the binary cross entropy loss with negative sampling, expressed by the formula:
                                        -∑_j log(x_ui) + log(1 - x_uj)
        where x_ui and x_uj are the prediction for user u on item i and j, respectively. Item i positive instance while
        Item j is a negative instance.

        :param logits: Logits values from the network.
        :param labels: 1-0 Labels.

        :return: The binary cross entropy as computed above
        """
        loss = nn.BCEWithLogitsLoss()(logits.flatten(), labels.flatten())

        return loss


class RecBayesianPersonalizedRankingLoss(RecommenderSystemLoss):

    def __init__(self):
        super().__init__()
        self.name = 'RecBayesianPersonalizedRankingLoss'

        logging.info(f'Built {self.name} module')

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        return RecBayesianPersonalizedRankingLoss()

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        It computes the Bayesian Personalized Ranking loss (https://arxiv.org/pdf/1205.2618.pdf).

        :param logits: Logits values from the network. The first column always contain the values of positive instances.
                Shape is (batch_size, 1 + n_neg).
        :param labels: 1-0 Labels. The first column contains 1s while all the others 0s.

        :return: The bayesian personalized ranking loss
        """
        pos_logits = logits[:, 0].unsqueeze(1)  # [batch_size,1]
        neg_logits = logits[:, 1:]  # [batch_size,n_neg]

        labels = labels[:, 0]  # I guess this is just to avoid problems with the device
        labels = torch.repeat_interleave(labels, neg_logits.shape[1])

        diff_logits = pos_logits - neg_logits

        loss = nn.BCEWithLogitsLoss()(diff_logits.flatten(), labels.flatten())

        return loss


class RecSampledSoftmaxLoss(RecommenderSystemLoss):
    """
    If the variables are passed to __init__ then the class adjust the loss computation by taking into account the sampling strategy, the # of total items, and the # of negative samples.
    See https://arxiv.org/pdf/2101.08769.pdf for more details
    """

    def __init__(self,
                 n_items: int = None,
                 train_neg_strategy: str = None,
                 neg_train: int = None):
        super().__init__()

        self.n_items = n_items
        self.train_neg_strategy = train_neg_strategy
        self.neg_train = neg_train

        self.name = 'RecSampledSoftmaxLoss'

        logging.info(f'Built {self.name} module')

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        return RecSampledSoftmaxLoss(n_items=dataset.n_items,
                                     train_neg_strategy=conf['train_neg_strategy'],
                                     neg_train=conf['neg_train'])

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        It computes the (Sampled) Softmax Loss (a.k.a. sampled cross entropy) expressed by the formula:
                            -x_ui +  ln( ∑_j e^{x_uj})
        where x_ui and x_uj are the prediction for user u on item i and j, respectively. Item i positive instance while j
        goes over all the sampled items (negatives + the positive). Negative samples are adjusted by a factor dependent
        on the sampling strategy and the number of sampled. I.e. ln( e^{x_ui} + ∑_{j!=i} e^{x_uj} * 1 /(n_neg * p(j|u))
        https://arxiv.org/pdf/2101.08769.pdf
        :param logits: Logits values from the network. The first column always contain the values of positive instances.
                Shape is (batch_size, 1 + n_neg).
        :param labels: 1-0 Labels. The first column contains 1s while all the others 0s.
        :return:
        """

        pos_logits_sum = - logits[:, 0]

        if self.train_neg_strategy == 'uniform':
            logits[:, 1:] += math.log(self.n_items / self.neg_train)
        log_sum_exp_sum = torch.logsumexp(logits, dim=-1)

        sampled_loss = pos_logits_sum + log_sum_exp_sum

        return sampled_loss.mean()


class DifferentiableRankingLoss(RecommenderSystemLoss):
    """
    It follows the implementation from "Differentiable Ranking Metric Using Relaxed Sorting for Top-K Recommendation" IEEE 2021
    https://ieeexplore.ieee.org/document/9514867
    """

    IMPLEMENTED_RANKING_METRIC = ['ndcg', 'precision']

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):

        assert logits.shape[
                   -1] < self.ranking_at_k, f"There are too few items to measure the1 ranking metric at k={self.ranking_at_k}"

        # Compute Metric-dependent weight
        rank_weight = self.weight_function(labels, k=self.ranking_at_k)  # [ranking_at_k]

        approx_sorting_matrix = self.neural_sort_layer(logits)  # [batch_size, n_scores (positions), n_scores (index) ]
        # approx_sorting_matrix[0] is the sorting matrix for the first user in the batch.
        # approx_sorting_matrix[0][0] gives us the one-hot representation of the item RANKED in the first position
        # e.g. logits[0] = [3,1,2] -> approx_sorting_matrix[0][0] = [1,0,0] (the one is selecting the 3)
        #                             approx_sorting_matrix[0][1] = [0,0,1] (the one is selecting the 2)
        #                             approx_sorting_matrix[0][2] = [0,1,0] (the one is selecting the 1)
        approx_sorting_matrix = approx_sorting_matrix[:, :self.ranking_at_k, :]

        weighted_sorting_matrix = (approx_sorting_matrix * rank_weight[:, None]).sum(
            1)  # [batch_size, n_scores (index)]

        mse_loss = self.mse_loss(weighted_sorting_matrix, labels, reduction='none')
        ranking_loss = mse_loss.mean(-1).mean()
        return ranking_loss

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        return DifferentiableRankingLoss(ranking_metric=conf['ranking_metric'],
                                         ranking_at_k=conf['ranking_at_k'],
                                         neural_sort_tau=conf['neural_sort_tau'],
                                         )

    def __init__(self, ranking_metric: str = 'ndcg', ranking_at_k: int = 10, neural_sort_tau: float = 1.):
        super().__init__()

        assert ranking_metric in DifferentiableRankingLoss.IMPLEMENTED_RANKING_METRIC, f'Ranking metric <{ranking_metric}> not implemented yet '
        assert neural_sort_tau > 0, 'Tau for NeuralSort should be positive! (ideally greater than 0)'

        self.ranking_metric = ranking_metric
        self.ranking_at_k = ranking_at_k
        self.neural_sort_tau = neural_sort_tau

        self.weight_function = None
        if self.ranking_metric == 'ndcg':
            self.weight_function = weight_ndcg_at_k_batch
        elif self.ranking_metric == 'precision':
            self.weight_function = weight_precision_at_k_batch
        else:
            raise NotImplementedError(f'Ranking metric <{self.ranking_metric}> not implemented yet ')

        self.neural_sort_layer = NeuralSort(self.neural_sort_tau)
        self.mse_loss = nn.MSELoss()

        self.name = 'DifferentiableRankingLoss'

        logging.info(f'Built {self.name} module'
                     f'- ranking_metric: {self.ranking_metric} \n'
                     f'- ranking_at_k: {self.ranking_at_k} \n'
                     f'- neural_sort_tau: {self.neural_sort_tau} \n')


class RecommenderSystemLossesEnum(Enum):
    bce = RecBinaryCrossEntropy
    bpr = RecBayesianPersonalizedRankingLoss
    sampled_softmax = RecSampledSoftmaxLoss
    drm = DifferentiableRankingLoss
