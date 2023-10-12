import logging
from abc import ABC
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import T_co, _worker_init_fn_t

from data.dataset import TrainRecDataset


class InteractionSampler(ABC):
    pass


class NegativeSampler(InteractionSampler):
    """
    NegativeSampler manages the parameters for negative sampling.
    """

    def __init__(self, train_dataset: TrainRecDataset, n_neg: int = 10, neg_sampling_strategy: str = 'uniform',
                 squashing_factor_pop_sampling: float = 1.):
        """
        :param train_dataset: TrainRecDataset. It should contain both pop_distribution and n_items.
        :param n_neg: Number of negative samples to take for each positive interaction
        :param neg_sampling_strategy: Either 'uniform' or 'popular'. See the respective functions for more details.
        :param squashing_factor_pop_sampling: Squashing factor for the popularity sampling. Ignored if neg_sampling_strategy = 'uniform'

        """
        assert n_neg > 0, 'Number of negatives should be > 0!'
        assert neg_sampling_strategy in ['uniform', 'popular'], \
            f'<{neg_sampling_strategy}> is not a valid negative sampling strategy!'
        assert squashing_factor_pop_sampling >= 0, 'Squashing factor for popularity sampling should be positive!'

        self.n_neg = n_neg
        self.neg_sampling_strategy = neg_sampling_strategy
        self.squashing_factor_pop_sampling = squashing_factor_pop_sampling

        self.n_items = train_dataset.n_items
        self.pop_distribution = train_dataset.pop_distribution.copy()

        if neg_sampling_strategy == 'uniform':
            self.neg_sampling_fun = self._neg_sample_uniform
        elif neg_sampling_strategy == 'popular':
            self.neg_sampling_fun = self._neg_sample_popular

        self.name = 'NegativeSampler'

        logging.info(f'Built {self.name} module \n'
                     f'- n_neg: {self.n_neg} \n'
                     f'- neg_sampling_strategy: {self.neg_sampling_strategy} \n'
                     f'- squashing_factor_pop_sampling: {self.squashing_factor_pop_sampling} \n')

    def _neg_sample_uniform(self, to_sample: int):
        return np.random.randint(0, high=self.n_items, size=to_sample)

    def _neg_sample_popular(self, to_sample: int):

        p = self.pop_distribution.copy()
        p = np.power(p, self.squashing_factor_pop_sampling)  # Applying squashing factor alpha
        p = p / p.sum()
        return np.random.choice(np.arange(self.n_items), size=to_sample, p=p)


class TrainDataLoader(DataLoader):
    """
    TrainDataLoader that performs negative sampling.
    """

    def __init__(self, interaction_sampler: InteractionSampler, dataset: Dataset[T_co], batch_size: Optional[int] = 1,
                 shuffle: bool = False,
                 sampler: Optional[Sampler] = None, batch_sampler: Optional[Sampler[Sequence]] = None,
                 num_workers: int = 0, pin_memory: bool = False,
                 drop_last: bool = False, timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None, *, prefetch_factor: int = 2,
                 persistent_workers: bool = False):

        self.interaction_sampler = interaction_sampler

        if isinstance(self.interaction_sampler, NegativeSampler):
            collate_function = self._neg_sampling_collate_fn
        else:
            raise ValueError('Invalid Interaction Sampler')

        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers,
                         collate_function, pin_memory,
                         drop_last, timeout, worker_init_fn, multiprocessing_context, generator,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

    def _neg_sampling_collate_fn(self, batch):
        """
        It performs the negative sampling procedure for the batch.
        @param batch: Batch is a list of tuples. Each tuple has the user_idx in [0] and item_idxs in [1].
        @return:
            -user_idxs. Tensor containing the user_idxs. Shape is [batch_size].
            - item_idxs. Tensor containing the item_idxs. Shape is [batch_size, n_pos (usually 1) + n_neg].
                The first columns holds the positive items
            - labels. Tensor containing the labels. Shape is [batch_size, n_pos (usually 1) + n_neg].
                The first columns holds the positive items

        """
        n_neg = self.interaction_sampler.n_neg
        batch_size = len(batch)
        user_idxs = np.array([x[0] for x in batch]).astype(np.int64)
        item_pos_idxs = np.array([x[1] for x in batch])
        n_pos = item_pos_idxs.shape[-1] if len(item_pos_idxs.shape) > 1 else 1

        item_neg_idxs = np.empty((batch_size, n_neg), dtype=np.int64)
        mask = np.ones_like(item_neg_idxs, dtype=bool)
        to_resample = mask.sum()

        while True:
            sampled_items = self.interaction_sampler.neg_sampling_fun(to_resample)
            item_neg_idxs[mask] = sampled_items

            for i in range(batch_size):
                mask[i] = np.isin(item_neg_idxs[i], self.dataset.sampling_matrix[user_idxs[i]].indices,
                                  assume_unique=True)
            to_resample = mask.sum()

            if to_resample == 0:
                break

        items_idxs = np.column_stack([item_pos_idxs, item_neg_idxs]).astype(np.int64)
        labels = np.zeros_like(items_idxs, dtype=float)
        labels[:, :n_pos] = 1.
        return torch.from_numpy(user_idxs), torch.from_numpy(items_idxs), torch.from_numpy(labels)
