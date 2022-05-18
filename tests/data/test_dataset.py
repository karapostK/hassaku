import unittest

import torch
from torch.utils.data import DataLoader

from data.dataset import FullEvalDataset, TrainRecDataset


class ML1MDatasetTest(unittest.TestCase):

    def setUp(self) -> None:
        self.batch_size = 10
        self.n_neg = 30

        self.train_dataset = TrainRecDataset('../../data/ml1m', n_neg=self.n_neg)
        self.val_dataset = FullEvalDataset('../../data/ml1m', 'val')
        self.test_dataset = FullEvalDataset('../../data/ml1m', 'test')

    def test_train_right_shape(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True)
        for user_idxs, item_idxs, labels in train_loader:
            self.assertEqual(user_idxs.shape, torch.Size([self.batch_size]))
            self.assertEqual(item_idxs.shape, torch.Size([self.batch_size, self.n_neg + 1]))
            self.assertEqual(labels.shape, torch.Size([self.batch_size, self.n_neg + 1]))

    def test_skip_0_users_eval(self):

        for eval_dataset in [self.val_dataset, self.test_dataset]:
            eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size)
            for _, _, labels in eval_loader:
                self.assertTrue((labels.sum(axis=-1) > 0).all())


if __name__ == '__main__':
    unittest.main()
