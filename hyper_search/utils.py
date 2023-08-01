import glob
import os
from typing import Dict, List

import numpy as np
from ray.tune import Callback
from ray.tune.experiment.trial import Trial


class KeepOnlyTopModels(Callback):
    """
    Callback class used to delete the models of the trials that are not in the top-n highest results.
    If the trial becomes part of the top-n, it deletes the model with the current smallest metric value among the top-n.
    This allows to keep the storage requirement low (e.g. 260 GB vs 16 GB for top-3 on ml10m)
    NB: In order to work properly, the file name should be something like "model" (e.g. model.npz or model.pth)
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

        print(f'Trial {trial.trial_id} ended with maximum metric: {round(trial_max, 4)}')
        print(f'Current top trial metrics: {[round(x, 4) for x in self._top_maxs]}')

        argmin = np.argmin(self._top_maxs)
        if self._top_maxs[argmin] < trial_max:
            print(f'Trial {trial.trial_id} became one of the top trials')
            # Save the current trial as current best

            old_trial_path = self._top_paths[argmin]
            self._top_maxs[argmin] = trial_max
            self._top_paths[argmin] = trial.logdir

            # Remove the previous-best
            # N.B. The class assumes that the model is stored in a model file!
            old_trial_checkpoint = os.path.join(old_trial_path, 'model.*')
            checkpoint_lists = glob.glob(old_trial_checkpoint)
            for checkpoint_file in checkpoint_lists:
                os.remove(checkpoint_file)

        else:
            print(f'Trial {trial.trial_id} did not become one of the top trials')
            # Delete self checkpoint
            trial_checkpoint = os.path.join(trial.logdir, 'model.*')
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
        best_path = self._top_paths[argmax]

        return best_value, best_path
