import os

import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm, trange

from algorithms.base_classes import SGDBasedRecommenderAlgorithm
from utilities.consts import OPTIMIZING_METRIC
from utilities.eval import Evaluator
from utilities.rec_losses import RecommenderSystemLossesEnum, RecommenderSystemLoss


class ExperimentConfig:
    """
    Helper class to keep all the configuration parameter in one place
    """

    def __init__(self, n_epochs: int = 100, device: str = 'cuda',
                 rec_loss: RecommenderSystemLoss = RecommenderSystemLossesEnum.bce.value(), lr: float = 1e-3,
                 wd: float = 1e-4, optim_type: 'str' = 'adam', max_patience: int = 10,
                 best_model_path: str = '../models/'):
        assert n_epochs > 0, f"Number of epochs ({n_epochs}) should be positive"
        assert device in ['cuda', 'cpu'], f"Device ({device}) not valid"
        assert lr > 0 and wd >= 0, f"Learning rate ({lr}) and Weight decay ({wd}) should be positive"
        assert optim_type in ['adam', 'adagrad'], f"Optimizer ({optim_type}) not implemented"
        assert max_patience is None or max_patience > 0, f"Max patience f({max_patience}) should either be None or positive"
        """
        :param max_patience: # of consecutive epochs in which the model does not improve over the validation data. After
        the number is reached, the training is halted.
        :param best_model_path: path whereto save the best model during training
        """

        self.n_epochs = n_epochs
        self.device = device
        self.rec_loss = rec_loss
        self.lr = lr
        self.wd = wd
        self.max_patience = max_patience
        self.best_model_path = best_model_path

        if optim_type == 'adam':
            self.optim = torch.optim.Adam
        elif optim_type == 'adagrad':
            self.optim = torch.optim.Adagrad


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

        self.model = model
        self.pointer_to_model = self.model
        if self.device == 'cuda':
            self.model = nn.DataParallel(self.model)
            self.pointer_to_model = self.model.module
        self.model.to(self.device)

        self.optimizing_metric = OPTIMIZING_METRIC
        self.max_patience = conf.max_patience
        self.best_model_path = os.path.join(conf.best_model_path, self.pointer_to_model.name + '_best_model.pth')

        self.optimizer = conf.optim(self.model.parameters(), lr=conf.lr, weight_decay=conf.wd)

        print(f'Built Trainer module')

    def fit(self):
        """
        Runs the Training procedure
        """

        metrics_values = self.val()
        best_value = metrics_values[self.optimizing_metric]
        # tune.report(metrics_values)
        print('Init - Avg Val Value {:.3f} \n'.format(best_value))

        patience = 0
        for epoch in trange(self.n_epochs):

            if patience == self.max_patience:
                print('Max Patience reached, stopping.')
                break

            self.model.train()

            epoch_train_loss = 0

            for u_idxs, i_idxs, labels in tqdm(self.train_loader):
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
            # tune.report({**metrics_values, 'epoch_train_loss': epoch_train_loss})

            if curr_value > best_value:
                best_value = curr_value
                print('Epoch {} - New best model found (val value {:.3f}) \n'.format(epoch, curr_value))
                torch.save(self.pointer_to_model.state_dict(), self.best_model_path)
                patience = 0
            else:
                patience += 1

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
        val_loss = 0
        evaluator = Evaluator(self.val_loader.dataset.n_users)

        for u_idxs, i_idxs, labels in self.val_loader:
            u_idxs = u_idxs.to(self.device)
            i_idxs = i_idxs.to(self.device)
            labels = labels.to(self.device)

            out = self.model(u_idxs, i_idxs)

            val_loss += self.rec_loss.compute_loss(out, labels).item()
            val_loss += self.pointer_to_model.get_and_reset_other_loss()

            out = out.to('cpu')

            evaluator.eval_batch(out)

        val_loss /= len(self.val_loader)
        metrics_values = {**evaluator.get_results(), 'val_loss': val_loss}

        return metrics_values
