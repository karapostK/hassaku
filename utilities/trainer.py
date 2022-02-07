import numpy as np
import torch
from ray import tune
from torch import nn
from torch.utils import data
from tqdm import trange

from algorithms.base_classes import SGDBasedRecommenderAlgorithm
from utilities.consts import OPTIMIZING_METRIC, SINGLE_SEED
from utilities.eval import evaluate_recommender_algorithm
from utilities.rec_losses import RecommenderSystemLossesEnum, RecommenderSystemLoss


class ExperimentConfig:
    """
    Helper class to keep all the configuration parameter in one place
    """

    def __init__(self, n_epochs: int = 100, device: str = 'cuda',
                 rec_loss: RecommenderSystemLoss = RecommenderSystemLossesEnum.bce.value(), lr: float = 1e-3,
                 wd: float = 1e-4, optim_type: 'str' = 'adam', best_model_path: str = './best_model.npz',
                 seed=SINGLE_SEED):

        assert n_epochs > 0, f"Number of epochs ({n_epochs}) should be positive"
        assert device in ['cuda', 'cpu'], f"Device ({device}) not valid"
        assert lr > 0 and wd >= 0, f"Learning rate ({lr}) and Weight decay ({wd}) should be positive"
        assert optim_type in ['adam', 'adagrad'], f"Optimizer ({optim_type}) not implemented"
        """
        :param best_model_path: path whereto save the best model during training
        """

        self.n_epochs = n_epochs
        self.device = device
        self.rec_loss = rec_loss
        self.lr = lr
        self.wd = wd
        self.best_model_path = best_model_path
        self.seed = seed

        if optim_type == 'adam':
            self.optim = torch.optim.Adam
        elif optim_type == 'adagrad':
            self.optim = torch.optim.Adagrad

    @staticmethod
    def build_from_conf(conf: dict):

        return ExperimentConfig(n_epochs=conf['n_epochs'],
                                rec_loss=conf['rec_loss'].value(),
                                lr=conf['optim_param']['lr'],
                                wd=conf['optim_param']['wd'],
                                optim_type=conf['optim_param']['optim'],
                                device=conf['device'] if 'device' in conf else 'cuda',
                                seed=conf['seed'])


class Trainer:

    def __init__(self, model: SGDBasedRecommenderAlgorithm, train_loader: data.DataLoader, val_loader: data.DataLoader,
                 conf: ExperimentConfig):
        """
        Train and Evaluate the model.
        :param model: Model to train
        :param train_loader: Training DataLoader
        :param val_loader: Validation DataLoader
        :param conf: Experiment configuration parameters
        """

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.n_epochs = conf.n_epochs
        self.rec_loss = conf.rec_loss

        self.device = conf.device
        self.seed = conf.seed

        self.model = model
        self.pointer_to_model = self.model
        if self.device == 'cuda':
            self.model = nn.DataParallel(self.model)
            self.pointer_to_model = self.model.module
        self.model.to(self.device)

        self.optimizing_metric = OPTIMIZING_METRIC
        self.best_model_path = conf.best_model_path

        self.optimizer = conf.optim(self.model.parameters(), lr=conf.lr, weight_decay=conf.wd)

        self.best_value = -np.inf

        print(f'Built Trainer module \n'
              f'- n_epochs: {self.n_epochs} \n'
              f'- rec_loss: {self.rec_loss.__class__.__name__} \n'
              f'- device: {self.device} \n'
              f'- optimizing_metric: {self.optimizing_metric} \n'
              f'- best_model_path: {self.best_model_path} \n')

    def fit(self):
        """
        Runs the Training procedure
        """

        metrics_values = self.val()
        self.best_value = metrics_values[self.optimizing_metric]

        print('Init - Avg Val Value {:.3f} \n'.format(self.best_value))

        for epoch in trange(self.n_epochs):

            self.model.train()

            epoch_train_loss = 0

            for u_idxs, i_idxs, labels in self.train_loader:
                u_idxs = u_idxs.to(self.device)
                i_idxs = i_idxs.to(self.device)
                labels = labels.to(self.device)

                out = self.model(u_idxs, i_idxs)

                loss = self.rec_loss.compute_loss(out, labels)
                loss += self.pointer_to_model.get_and_reset_other_loss()

                epoch_train_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_train_loss /= len(self.train_loader)
            print("Epoch {} - Epoch Avg Train Loss {:.3f} \n".format(epoch, epoch_train_loss))

            metrics_values = self.val()
            curr_value = metrics_values[self.optimizing_metric]
            print('Epoch {} - Avg Val Value {:.3f} \n'.format(epoch, curr_value))
            tune.report(**{**metrics_values, 'epoch_train_loss': epoch_train_loss})

            if curr_value > self.best_value:
                self.best_value = curr_value
                print('Epoch {} - New best model found (val value {:.3f}) \n'.format(epoch, curr_value))
                self.pointer_to_model.save_model_to_path(self.best_model_path)

        # Fitting is over, return the best model
        trained_model = self.pointer_to_model.to('cpu')
        params = torch.load(self.best_model_path, map_location='cpu')
        trained_model.load_state_dict(params)
        return trained_model

    @torch.no_grad()
    def val(self):
        """
        Runs the evaluation procedure.
        :return: the dictionary of the metric values
        """
        self.model.eval()
        print('Validation started')

        metrics_values = evaluate_recommender_algorithm(self.pointer_to_model, self.val_loader, self.seed + 1, self.device,
                                                        self.rec_loss)
        return metrics_values
