import glob
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
from ray.tune import Stopper, Callback


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
