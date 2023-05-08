import unittest
import torch
import math

from eval.metrics import recall_at_k_batch, precision_at_k_batch, ndcg_at_k_batch


class TestMetrics(unittest.TestCase):

    def setUp(self) -> None:
        self.batch_size = 10
        self.n_items = 20
        self.k = 10

        self.logits = torch.arange(self.n_items, 0, -1).repeat(self.batch_size, 1)
        self.y_true_zeros = torch.zeros((self.batch_size, self.n_items))
        self.y_true_ones = torch.ones((self.batch_size, self.n_items))

        self.y_true_1 = torch.zeros((self.batch_size, self.n_items))
        self.y_true_1[:, 0] = 1

        self.y_true_2_and_3 = torch.zeros((self.batch_size, self.n_items))
        self.y_true_2_and_3[:, [1, 2]] = 1

        self.y_true_out_of_k = torch.zeros((self.batch_size, self.n_items))
        self.y_true_out_of_k[:, self.k + 1:] = 1
        self.y_true_out_of_k[:, 0] = 1

    def test_recall(self):
        recall_zeros = recall_at_k_batch(self.logits, self.y_true_zeros, k=self.k).item() / self.batch_size
        recall_ones = recall_at_k_batch(self.logits, self.y_true_ones, k=self.k).item() / self.batch_size
        recall_1 = recall_at_k_batch(self.logits, self.y_true_1, k=self.k).item() / self.batch_size
        recall_2_and_3 = recall_at_k_batch(self.logits, self.y_true_2_and_3, k=self.k).item() / self.batch_size
        recall_out_of_k = recall_at_k_batch(self.logits, self.y_true_out_of_k, k=self.k).item() / self.batch_size

        self.assertEqual(recall_zeros, 0)
        self.assertAlmostEqual(recall_ones, self.k / self.n_items)
        self.assertEqual(recall_1, 1)
        self.assertEqual(recall_2_and_3, 1)
        self.assertAlmostEqual(recall_out_of_k, 1 / (self.n_items - self.k))

    def test_precision(self):
        precision_zeros = precision_at_k_batch(self.logits, self.y_true_zeros, k=self.k).item() / self.batch_size
        precision_ones = precision_at_k_batch(self.logits, self.y_true_ones, k=self.k).item() / self.batch_size
        precision_1 = precision_at_k_batch(self.logits, self.y_true_1, k=self.k).item() / self.batch_size
        precision_2_and_3 = precision_at_k_batch(self.logits, self.y_true_2_and_3, k=self.k).item() / self.batch_size
        precision_out_of_k = precision_at_k_batch(self.logits, self.y_true_out_of_k, k=self.k).item() / self.batch_size

        self.assertEqual(precision_zeros, 0)
        self.assertEqual(precision_ones, 1)
        self.assertAlmostEqual(precision_1, 1 / self.k)
        self.assertAlmostEqual(precision_2_and_3, 2 / self.k)
        self.assertAlmostEqual(precision_out_of_k, 1 / self.k)

    def test_ndcg(self):
        ndcg_zeros = ndcg_at_k_batch(self.logits, self.y_true_zeros, k=self.k).item() / self.batch_size
        ndcg_ones = ndcg_at_k_batch(self.logits, self.y_true_ones, k=self.k).item() / self.batch_size
        ndcg_1 = ndcg_at_k_batch(self.logits, self.y_true_1, k=self.k).item() / self.batch_size
        ndcg_2_and_3 = ndcg_at_k_batch(self.logits, self.y_true_2_and_3, k=self.k).item() / self.batch_size
        ndcg_out_of_k = ndcg_at_k_batch(self.logits, self.y_true_out_of_k, k=self.k).item() / self.batch_size

        discount_template = 1. / torch.log2(torch.arange(2, self.k + 2).float())

        self.assertEqual(ndcg_zeros, 0)
        self.assertEqual(ndcg_ones, 1)
        self.assertEqual(ndcg_1, 1)
        self.assertAlmostEqual(ndcg_2_and_3, (math.log2(4) + math.log2(3)) / (math.log2(4) * (1 + math.log2(3))),
                               delta=1e-5)
        self.assertAlmostEqual(ndcg_out_of_k, 1 / discount_template[:min(self.k, self.n_items - self.k)].sum().item())


if __name__ == '__main__':
    unittest.main()
