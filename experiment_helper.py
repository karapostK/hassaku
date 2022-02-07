import json
import os
import shutil
from collections import defaultdict

import numpy as np
import wandb
from filelock import FileLock
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import CombinedStopper, TrialPlateauStopper
from ray.tune.suggest.hyperopt import HyperOptSearch

from algorithms.base_classes import SGDBasedRecommenderAlgorithm
from data.dataset import get_recdataset_dataloader
from hyper_params import alg_param
from utilities.consts import NEG_VAL, SINGLE_SEED, PROJECT_NAME, WANDB_API_KEY_PATH, DATA_PATH, OPTIMIZING_METRIC, \
    SEED_LIST
from utilities.enums import RecAlgorithmsEnum, RecDatasetsEnum
from utilities.eval import evaluate_recommender_algorithm
from utilities.trainer import ExperimentConfig, Trainer
from utilities.utils import generate_id, reproducible, NoImprovementsStopper


def load_data(conf: dict, split_set: str):
    if split_set == 'train':
        train_loader = get_recdataset_dataloader(
            'inter',
            data_path=conf['data_path'],
            split_set='train',
            n_neg=conf['neg_train'] if "neg_train" in conf else 4,
            neg_strategy=conf['train_neg_strategy'] if 'train_neg_strategy' in conf else 'uniform',
            batch_size=conf['batch_size'] if 'batch_size' in conf else 64,
            shuffle=True,
            num_workers=2
        )
        return train_loader
    elif split_set == 'val':
        val_loader = get_recdataset_dataloader(
            'inter',
            data_path=conf['data_path'],
            split_set='val',
            n_neg=NEG_VAL,
            neg_strategy=conf['eval_neg_strategy'],
            batch_size=conf['val_batch_size'],
        )
        return val_loader
    elif split_set == 'test':
        test_loader = get_recdataset_dataloader(
            'inter',
            data_path=conf['data_path'],
            split_set='test',
            n_neg=NEG_VAL,
            neg_strategy=conf['eval_neg_strategy'],
            batch_size=conf['val_batch_size'],
        )

        return test_loader


def check_whether_to_save(run_metric: float, checkpoint_file: str) -> bool:
    """
    This function manages the saving of the found models by saving only the top-3 best models for a specific algorithm.
    This avoids saving all NUM_SAMPLES configurations of models.
    If the trial becomes part of the top-3, it deletes the model with the current smallest metric value among the top-3. It returns to_save=True
    If the trial does not become part of the top-3, do nothing and return to_save=False
    :param run_metric: the metric of the current trial which will be used to check whether we save the model or not
    :param checkpoint_file: the path pointing at the model
    :return: boolean value indicating whether to save the current model
    """

    LOCK_PATH = '../file.loc'
    SYNC_DATA_PATH = '../sync_data.json'

    with FileLock(LOCK_PATH):

        # Create file if not there the first time
        if not os.path.isfile(SYNC_DATA_PATH):
            with open(SYNC_DATA_PATH, 'w') as out_file:
                json.dump({'paths': [''] * 3, 'values': [-np.inf] * 3}, out_file)

        # Read the file used for synchronization
        with open(SYNC_DATA_PATH, 'r') as in_file:
            sync_data = json.load(in_file)
            top_paths = sync_data['paths']
            top_values = sync_data['values']

        # Compare the current trial with the trial that has the minimum metric value within the top-3
        argmin = np.argmin(top_values)
        if top_values[argmin] < run_metric:
            # Save
            print('--- Run saved ---')
            print('This trial: ', round(run_metric, 3))
            print('Current top-3: ', [round(x, 3) for x in top_values])

            # Delete previous trial
            old_path = top_paths[argmin]
            if os.path.isdir(old_path):
                shutil.rmtree(old_path)

            top_values[argmin] = run_metric
            top_paths[argmin] = checkpoint_file

            with open(SYNC_DATA_PATH, 'w') as out_file:
                json.dump({'paths': top_paths, 'values': top_values}, out_file)
            return True
        else:
            # Don't save
            return False


def tune_training(conf: dict, checkpoint_dir=None):
    """
    Function executed by ray tune. It corresponds to a single trial.
    """

    train_loader = load_data(conf, 'train')
    val_loader = load_data(conf, 'val')

    reproducible(conf['seed'])

    alg = conf['alg'].value.build_from_conf(conf, train_loader.dataset)
    with tune.checkpoint_dir(0) as checkpoint_dir:

        if isinstance(alg, SGDBasedRecommenderAlgorithm):

            checkpoint_file = os.path.join(checkpoint_dir, 'best_model.pth')
            exp_conf = ExperimentConfig.build_from_conf(conf)
            exp_conf.best_model_path = checkpoint_file

            trainer = Trainer(alg, train_loader, val_loader, exp_conf)
            trainer.fit()

            to_save = check_whether_to_save(trainer.best_value, checkpoint_file)
            if not to_save and os.path.isfile(checkpoint_file):
                # Delete self since it shouldn't have been saved
                os.remove(checkpoint_file)

        else:
            checkpoint_file = os.path.join(checkpoint_dir, 'best_model.npz')
            # -- Training --
            # todo: ensure that all algorithms have the following method!
            alg.fit(train_loader.dataset.iteration_matrix)

            # -- Validation --
            metrics_values = evaluate_recommender_algorithm(alg, val_loader, conf['seed'] + 1)
            tune.report(**metrics_values)

            # -- Save --
            run_metric = metrics_values[OPTIMIZING_METRIC]

            to_save = check_whether_to_save(run_metric, checkpoint_file)

            # Save
            if to_save:
                alg.save_model_to_path(checkpoint_file)


def run_train_val(conf: dict, run_name: str, **kwargs):
    """
    Runs the train and validation procedure.
    """

    # Hyperparameter Optimization
    metric_name = OPTIMIZING_METRIC

    # Search Algorithm
    search_alg = HyperOptSearch(random_state_seed=conf['seed'])

    if os.path.basename(conf['data_path']) == 'lfm2b1m':
        scheduler = ASHAScheduler(grace_period=4)
    else:
        scheduler = None

    # Logger
    callback = WandbLoggerCallback(project=PROJECT_NAME, log_config=True, api_key_file=WANDB_API_KEY_PATH,
                                   reinit=True, force=True, job_type='train/val', tags=run_name.split('_'))

    # Stopper
    stopper = CombinedStopper(
        NoImprovementsStopper(metric_name, max_patience=10),
        TrialPlateauStopper(metric_name, std=1e-3, num_results=5, grace_period=10)
    )

    tune.register_trainable(run_name, tune_training)
    analysis = tune.run(
        run_name,
        config=conf,
        name=generate_id(prefix=run_name),
        resources_per_trial={'gpu': kwargs['n_gpus']},
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=kwargs['n_samples'],
        callbacks=[callback],
        metric=metric_name,
        stop=stopper,
        max_concurrent_trials=kwargs['n_concurrent'],
        mode='max',
        fail_fast=True,
    )
    best_trial = analysis.get_best_trial(metric_name, 'max', scope='all')
    best_config = best_trial.config
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric_name, 'max')

    print('Train and Val ended')
    print(f'Best configuration is: \n {best_config}')
    print(f'Best checkpoint is: \n {best_checkpoint}')

    return best_config, best_checkpoint


def run_test(run_name: str, best_config: dict, best_checkpoint=''):
    """
    Runs the test procedure. Notice that the evaluation of the recommendation uses conf['seed'] + 2!
    """
    test_loader = load_data(best_config, 'test')

    wandb.login()
    wandb.init(project=PROJECT_NAME, group='test', config=best_config, name=run_name, force=True,
               job_type='test', tags=run_name.split('_'))

    # ---- Test ---- #
    alg = best_config['alg'].value.build_from_conf(best_config, test_loader.dataset)

    best_checkpoint = os.path.join(best_checkpoint, 'best_model.npz')
    alg.load_model_from_path(best_checkpoint)
    metrics_values = evaluate_recommender_algorithm(alg, test_loader, best_config['seed'] + 2)
    wandb.log(metrics_values)

    wandb.finish()

    return metrics_values


def start_hyper(alg: RecAlgorithmsEnum, dataset: RecDatasetsEnum, seed: int = SINGLE_SEED, **kwargs) -> dict:
    print('Starting Hyperparameter Optimization')
    print(f'Seed is {seed}')
    print('N.B. Val seed is obtained by seed + 1, Test seed by seed + 2 ')

    # ---- Algorithm's parameters and hyperparameters ---- #
    conf = alg_param[alg]
    conf['alg'] = alg

    # Dataset
    conf['data_path'] = os.path.join(os.getcwd(), DATA_PATH, dataset.name)

    # Seed
    conf['seed'] = seed

    # Hostname
    host_name = os.uname()[1][:2]

    run_name = f'{alg.name}_{dataset.name}_{host_name}_{seed}'

    # ---- Train/Validation ---- #
    print('Start Train/Val')
    best_config, best_checkpoint = run_train_val(conf, run_name, **kwargs)

    print('Start Test')
    metric_values = run_test(run_name, best_config, best_checkpoint)

    print('End')
    return metric_values


def start_multiple_hyper(alg: RecAlgorithmsEnum, dataset: RecDatasetsEnum, **kwargs):
    print('Starting Multi-Hyperparameter Optimization')
    print(f'Seeds are {SEED_LIST}')

    # Accumulate the results in a dictionary: e.g. results_list['ndcg@10'] = [0.8,0.5,0.3]
    results_dict = defaultdict(list)

    # Carry out the experiment
    for seed in SEED_LIST:
        metric_values = start_hyper(alg, dataset, seed, **kwargs)
        for key in metric_values:
            results_dict[key].append(metric_values[key])

    # Having collected all the values, carry out mean and std
    aggr_results_dict = defaultdict(lambda x: 0)
    for key, values_list in results_dict.items():
        aggr_results_dict[f'{key} mu'] = np.mean(values_list)
        aggr_results_dict[f'{key} sig'] = np.std(values_list)

    # Log results
    with open(WANDB_API_KEY_PATH) as wandb_file:
        wandb_api_key = wandb_file.read()

    run_name = f'{alg.name}_{dataset}'
    wandb.login(key=wandb_api_key)
    wandb.init(project=PROJECT_NAME, group='aggr_results', name=run_name, force=True, job_type='test',
               tags=run_name.split('_'))

    wandb.log(aggr_results_dict)
    wandb.finish()


def start_multi_dataset(alg: RecAlgorithmsEnum, **kwargs):
    print('Starting Multi-dataset Multi-Hyperparameter Optimization')
    for dataset in RecDatasetsEnum:
        start_multiple_hyper(alg, dataset, **kwargs)
