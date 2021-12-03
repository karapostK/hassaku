import torch
from torch import nn
from torch.utils import data

from algorithms.base_classes import SGDBasedRecommenderAlgorithm
from utilities.eval import Evaluator
from utilities.trainer import ExperimentConfig
from utilities.utils import print_results


class Tester:

    def __init__(self, model: SGDBasedRecommenderAlgorithm, test_loader: data.DataLoader, conf: ExperimentConfig):
        """
        Test the model
        :param model: Model to test
        :param test_loader: Test DataLoader
        :param conf: Experiment configuration parameters
        """

        self.test_loader = test_loader

        self.rec_loss = conf.rec_loss

        self.device = conf.device

        self.model = model
        self.pointer_to_model = self.model
        if self.device == 'cuda':
            self.model = nn.DataParallel(self.model)
            self.pointer_to_model = self.model.module
        self.model.to(self.device)

        print(f'Built Tester module')

    @torch.no_grad()
    def test(self):
        """
        Runs the evaluation procedure.
        """

        self.model.eval()
        print('Testing started')
        test_loss = 0
        evaluator = Evaluator(self.test_loader.dataset.n_users)

        for u_idxs, i_idxs, labels in self.test_loader:
            u_idxs = u_idxs.to(self.device)
            i_idxs = i_idxs.to(self.device)
            labels = labels.to(self.device)

            out = self.model(u_idxs, i_idxs)

            test_loss += self.rec_loss.compute_loss(out, labels).item()
            test_loss += self.pointer_to_model.get_and_reset_other_loss()

            out = out.to('cpu')

            evaluator.eval_batch(out)

        test_loss /= len(self.test_loader)

        metrics_values = {**evaluator.get_results(), 'test_loss': test_loss}

        print_results(metrics_values)

        return metrics_values
