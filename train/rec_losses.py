import math
from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import nn


class RecommenderSystemLoss(ABC):
    def __init__(self, n_items: int = None, aggregator: str = 'mean', train_neg_strategy: str = 'uniform',
                 neg_train: int = 4):
        assert aggregator in ['mean', 'sum'], "Type of Aggregator not yet defined"
        assert train_neg_strategy in ['uniform'], "Type of Negative Strategy not currently supported"

        super().__init__()
        self.n_items = n_items
        self.aggregator = aggregator
        self.train_neg_strategy = train_neg_strategy
        self.neg_train = neg_train

    @abstractmethod
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        pass

    @staticmethod
    def build_from_conf(conf: dict, dataset):
        rec_loss_class = RecommenderSystemLossesEnum[conf['rec_loss']]
        return rec_loss_class.value(
            n_items=dataset.n_items,
            aggregator=conf['loss_aggregator'],
            train_neg_strategy=conf['train_neg_strategy'],
            neg_train=conf['neg_train']
        )


class RecBinaryCrossEntropy(RecommenderSystemLoss):
    # Todo: this loss should be adjusted according to the sampling probability

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        It computes the binary cross entropy loss with negative sampling, expressed by the formula:
                                        -∑_j log(x_ui) + log(1 - x_uj)
        where x_ui and x_uj are the prediction for user u on item i and j, respectively. Item i positive instance while
        Item j is a negative instance.

        :param logits: Logits values from the network. The first column always contain the values of positive instances.
                Shape is (batch_size, 1 + n_neg).
        :param labels: 1-0 Labels. The first column contains 1s while all the others 0s.

        :return: The binary cross entropy as computed above
        """
        loss = nn.BCEWithLogitsLoss(reduction=self.aggregator)(logits.flatten(), labels.flatten())

        return loss


class RecBayesianPersonalizedRankingLoss(RecommenderSystemLoss):

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

        loss = nn.BCEWithLogitsLoss(reduction=self.aggregator)(diff_logits.flatten(), labels.flatten())

        return loss


class RecSampledSoftmaxLoss(RecommenderSystemLoss):

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

        if self.aggregator == 'sum':
            return sampled_loss.sum()
        elif self.aggregator == 'mean':
            return sampled_loss.mean()


class RecommenderSystemLossesEnum(Enum):
    bce = RecBinaryCrossEntropy
    bpr = RecBayesianPersonalizedRankingLoss
    sampled_softmax = RecSampledSoftmaxLoss
