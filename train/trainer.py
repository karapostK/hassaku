import logging

import torch
import wandb
from ray.air import session
from torch import nn
from torch.utils import data
from tqdm import trange, tqdm

from algorithms.base_classes import SGDBasedRecommenderAlgorithm
from eval.eval import evaluate_recommender_algorithm
from train.rec_losses import RecommenderSystemLossesEnum


class Trainer:

    def __init__(self, model: SGDBasedRecommenderAlgorithm,
                 train_loader: data.DataLoader,
                 val_loader: data.DataLoader,
                 conf: dict):
        """
        Train and Evaluate the model.
        :param model: Model to train
        :param train_loader: Training DataLoader
        :param val_loader: Validation DataLoader
        :param conf: Configuration dictionary
        """

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = conf['device']

        self.model = model
        self.pointer_to_model = self.model
        if self.device == 'cuda':
            self.model = nn.DataParallel(self.model)
            self.pointer_to_model = self.model.module
        self.model.to(self.device)

        self.rec_loss = RecommenderSystemLossesEnum[conf['rec_loss']].value(n_items=train_loader.dataset.n_items,
                                                                            aggregator=conf['loss_aggregator'],
                                                                            train_neg_strategy=conf[
                                                                                'train_neg_strategy'],
                                                                            neg_train=conf['neg_train'])

        self.lr = conf['lr']
        self.wd = conf['wd']

        if conf['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        elif conf['optimizer'] == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        elif conf['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            raise ValueError(f"Optimizer {conf['optimizer']} not yet implemented")

        self.n_epochs = conf['n_epochs']
        self.optimizing_metric = conf['optimizing_metric']
        self.max_patience = conf['max_patience']

        self.model_path = conf['model_path']

        running_settings = conf['running_settings']
        self.use_wandb = running_settings['use_wandb']
        self.batch_verbose = running_settings['batch_verbose']

        self._in_tune = conf['_in_tune'] if '_in_tune' in conf else False

        self.best_value = None
        self.best_metrics = None
        self.best_epoch = None
        logging.info(f'Built Trainer module \n'
                     f'- n_epochs: {self.n_epochs} \n'
                     f'- rec_loss: {self.rec_loss.__class__.__name__} \n'
                     f'- loss_aggregator: {self.rec_loss.aggregator} \n'
                     f'- device: {self.device} \n'
                     f'- optimizing_metric: {self.optimizing_metric} \n'
                     f'- model_path: {self.model_path} \n'
                     f'- optimizer: {self.optimizer.__class__.__name__} \n'
                     f'- lr: {self.lr} \n'
                     f'- wd: {self.wd} \n'
                     f'- use_wandb: {self.use_wandb} \n'
                     f'- batch_verbose: {self.batch_verbose} \n'
                     f'- max_patience: {self.max_patience} \n')

    def fit(self):
        """
        Runs the Training procedure
        """

        current_patience = self.max_patience

        metrics_values = self.val()

        self.best_value = metrics_values['max_optimizing_metric'] = metrics_values[self.optimizing_metric]
        self.best_epoch = metrics_values['best_epoch'] = -1
        self.best_metrics = metrics_values

        print('Init - Avg Val Value {:.3f} \n'.format(self.best_value))

        if self.use_wandb and not self._in_tune:
            wandb.log(metrics_values)

        if self._in_tune:
            session.report(metrics_values)

        self.pointer_to_model.save_model_to_path(self.model_path)

        for epoch in trange(self.n_epochs):

            self.model.train()

            if current_patience == 0:
                print('Ran out of patience, Stopping ')
                break

            epoch_losses = {
                'epoch_train_loss': 0,
                'epoch_train_rec_loss': 0,
            }

            if self.batch_verbose:
                iterator = tqdm(self.train_loader)
            else:
                iterator = self.train_loader

            for u_idxs, i_idxs, labels in iterator:
                u_idxs = u_idxs.to(self.device)
                i_idxs = i_idxs.to(self.device)
                labels = labels.to(self.device)

                out = self.model(u_idxs, i_idxs)

                rec_loss = self.rec_loss.compute_loss(out, labels)
                reg_losses = self.pointer_to_model.get_and_reset_other_loss()
                reg_loss = reg_losses['reg_loss'].to(rec_loss.device)

                total_loss = rec_loss + reg_loss

                epoch_losses['epoch_train_loss'] += total_loss.item()
                epoch_losses['epoch_train_rec_loss'] += rec_loss.item()
                reg_losses = {'epoch_train_' + k: v.item() for k, v in reg_losses.items()}
                epoch_losses.update({k: reg_losses[k] + epoch_losses.get(k, 0) for k in reg_losses})

                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_losses = {k: v / len(self.train_loader) for k, v in epoch_losses.items()}

            print("Epoch {} - Epoch Avg Train Loss {:.3f} ({:.3f} Rec Loss + {:.3f} Reg Loss )\n".
                  format(epoch, epoch_losses['epoch_train_loss'], epoch_losses['epoch_train_rec_loss'],
                         epoch_losses['epoch_train_reg_loss']))

            metrics_values = self.val()
            curr_value = metrics_values[self.optimizing_metric]
            print('Epoch {} - Avg Val Value {:.3f} \n'.format(epoch, curr_value))

            if curr_value > self.best_value:
                self.best_value = metrics_values['max_optimizing_metric'] = curr_value
                self.best_epoch = metrics_values['best_epoch'] = epoch
                self.best_metrics = metrics_values

                print('Epoch {} - New best model found (val value {:.3f}) \n'.format(epoch, curr_value))
                self.pointer_to_model.save_model_to_path(self.model_path)

                current_patience = self.max_patience  # Reset patience
            else:
                metrics_values['max_optimizing_metric'] = self.best_value
                current_patience -= 1

            # Logging
            log_dict = {**metrics_values, **epoch_losses}
            # Execute a post validation function that is specific to the model
            if hasattr(self.pointer_to_model, 'post_val') and callable(self.pointer_to_model.post_val):
                log_dict.update(self.pointer_to_model.post_val(epoch))

            if self.use_wandb and not self._in_tune:
                wandb.log(log_dict)

            if self._in_tune:
                session.report(log_dict)

        return self.best_metrics

    @torch.no_grad()
    def val(self):
        """
        Runs the evaluation procedure.
        :return: the dictionary of the metric values
        """
        self.model.eval()
        print('Validation started')

        metrics_values = evaluate_recommender_algorithm(self.pointer_to_model, self.val_loader, self.device)
        return metrics_values
