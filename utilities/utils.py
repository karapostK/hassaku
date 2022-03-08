import functools
import glob
import json
import os
import pickle
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from ray.tune import Stopper, Callback
from torch import nn


def pickle_load(file_path):
    return pickle.load(open(file_path, 'rb'))


def pickle_dump(x, file_path):
    return pickle.dump(x, open(file_path, 'wb'))


def json_load(file_path):
    return json.load(open(file_path, 'r'))


def general_weight_init(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        if m.weight.requires_grad:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                torch.nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Embedding:
        if m.weight.requires_grad:
            torch.nn.init.normal_(m.weight)

    elif type(m) == nn.BatchNorm2d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def generate_id(prefix=None, postfix=None):
    dateTimeObj = datetime.now()
    uid = '{}-{}-{}_{}-{}-{}.{}'.format(dateTimeObj.year, dateTimeObj.month, dateTimeObj.day, dateTimeObj.hour,
                                        dateTimeObj.minute, dateTimeObj.second, dateTimeObj.microsecond)
    if not prefix is None:
        uid = prefix + "_" + uid
    if not postfix is None:
        uid = uid + "_" + postfix
    return uid


def reproducible(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_results(metrics):
    """
    Prints the results on the command line.

    :param metrics: dict containing the metrics to print.
    :return:
    """

    STR_RESULT = "{:10} : {:.3f}"

    for metric_name, metric_value in metrics.items():
        print(STR_RESULT.format(metric_name, metric_value))


def generate_slices(total_columns):
    """
    Generate slices that will be processed based on the number of cores
    available on the machine.
    """
    from multiprocessing import cpu_count

    cores = cpu_count()
    print('Running on {} cores'.format(cores))
    segment_length = total_columns // cores

    ranges = []
    now = 0

    while now < total_columns:
        end = now + segment_length

        # The last part can be a little greater that others in some cases, but
        # we can't generate more than #cores ranges
        end = end if end + segment_length <= total_columns else total_columns
        ranges.append((now, end))
        now = end

    return ranges


class FunctionWrapper:
    """
    Since functions are not properly recognized as enum items, we need to use a wrapper function.
    """

    def __init__(self, function):
        self.function = function
        functools.update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __repr__(self):
        return self.function.__repr__()


class NoImprovementsStopper(Stopper):

    def __init__(self, metric: str, max_patience: int = 10):
        self.metric = metric
        self.max_patience = max_patience

        self._curr_patience_dict = defaultdict(lambda: self.max_patience)
        self._curr_max_dict = defaultdict(lambda: -np.inf)

    def __call__(self, trial_id, result):
        current_metric = result.get(self.metric)

        if current_metric > self._curr_max_dict[trial_id]:
            self._curr_patience_dict[trial_id] = self.max_patience
            self._curr_max_dict[trial_id] = current_metric
        else:
            self._curr_patience_dict[trial_id] -= 1

        if self._curr_patience_dict[trial_id] == 0:
            print('Maximum Patience Reached. Stopping')
            return True
        else:
            return False

    def stop_all(self):
        return False


class KeepOnlyTopTrials(Callback):
    """
    Callback class used to keep only the trials with the highest results.
    If the trial becomes part of the top-n, it deletes the model with the current smallest metric value among the top-n.
    """

    def __init__(self, metric_name: str, n_tops: int = 3):
        """
        :param metric_name: metric used to compare trials among themselves
        :param n_tops: how many trials to keep saved. Default to keeping only the top-3
        """
        self.metric_name = metric_name
        self._trials_maxs: Dict["Trial", float] = {}

        self._top_maxs: List[float] = [-np.inf] * n_tops
        self._top_paths: List[str] = [''] * n_tops
        self._top_confs: List[dict] = [{}] * n_tops

    def on_trial_result(self, iteration: int, trials: List["Trial"],
                        trial: "Trial", result: Dict, **info):
        current_max = self._trials_maxs.get(trial, -np.inf)

        if current_max < result[self.metric_name]:
            # Update own max
            current_max = result[self.metric_name]
            self._trials_maxs[trial] = current_max

    def on_trial_complete(self, iteration: int, trials: List["Trial"],
                          trial: "Trial", **info):

        trial_max = self._trials_maxs[trial]

        print(f'Trial {trial.trial_id} ended with maximum metric: {round(trial_max, 3)}')
        print(f'Current top trial metrics: {[round(x, 3) for x in self._top_maxs]}')

        argmin = np.argmin(self._top_maxs)
        if self._top_maxs[argmin] < trial_max:
            print(f'Trial {trial.trial_id} became one of the top trials')
            # Save the current trial as current best

            old_trial_path = self._top_paths[argmin]
            self._top_maxs[argmin] = trial_max
            self._top_paths[argmin] = trial.logdir
            self._top_confs[argmin] = trial.config

            # Remove the previous-best
            # N.B. The framework assumes that there is only a single checkpoint!
            old_trial_checkpoint = os.path.join(old_trial_path, 'checkpoint_000000/best*')
            checkpoint_lists = glob.glob(old_trial_checkpoint)
            for checkpoint_file in checkpoint_lists:
                os.remove(checkpoint_file)

        else:
            print(f'Trial {trial.trial_id} did not become one of the top trials')
            # Delete self checkpoint
            trial_checkpoint = os.path.join(trial.logdir, 'checkpoint_000000/best*')
            checkpoint_lists = glob.glob(trial_checkpoint)
            for checkpoint_file in checkpoint_lists:
                os.remove(checkpoint_file)

    def get_best_trial(self):
        """
        Get the values, configuration, and path of the best model
        NB. This method should be called only at the end of all experiments!
        """
        argmax = np.argmax(self._top_maxs)

        best_value = self._top_maxs[argmax]
        best_path = os.path.join(self._top_paths[argmax], 'checkpoint_000000')
        best_conf = self._top_confs[argmax]

        return best_value, best_path, best_conf
