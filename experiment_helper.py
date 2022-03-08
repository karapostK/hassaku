import os
from collections import defaultdict

import numpy as np
import wandb
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
from utilities.eval import evaluate_recommender_algorithm, evaluate_naive_algorithm
from utilities.trainer import ExperimentConfig, Trainer
from utilities.utils import generate_id, reproducible, NoImprovementsStopper, KeepOnlyTopTrials


def load_data(conf: dict, split_set: str, **kwargs):
    if split_set == 'train':
        train_loader = get_recdataset_dataloader(
            'inter',
            data_path=conf['data_path'],
            split_set='train',
            n_neg=conf['neg_train'] if "neg_train" in conf else 4,
            neg_strategy=conf['train_neg_strategy'] if 'train_neg_strategy' in conf else 'uniform',
            batch_size=conf['batch_size'] if 'batch_size' in conf else 64,
            shuffle=True,
            num_workers=kwargs['n_workers']
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


def tune_training(conf: dict, checkpoint_dir=None):
    """
    Function executed by ray tune. It corresponds to a single trial.
    """

    train_loader = load_data(conf, 'train', n_workers=conf['experiment_settings']['n_workers'])
    val_loader = load_data(conf, 'val')

    reproducible(conf['seed'])

    alg = conf['alg'].value.build_from_conf(conf, train_loader.dataset)
    with tune.checkpoint_dir(0) as checkpoint_dir:

        if isinstance(alg, SGDBasedRecommenderAlgorithm):

            checkpoint_file = os.path.join(checkpoint_dir, 'best_model.pth')
            conf['n_epochs'] = conf['experiment_settings']['n_epochs']
            exp_conf = ExperimentConfig.build_from_conf(conf)
            exp_conf.best_model_path = checkpoint_file

            # Validation happens within Trainer
            trainer = Trainer(alg, train_loader, val_loader, exp_conf)
            trainer.fit()

        else:
            checkpoint_file = os.path.join(checkpoint_dir, 'best_model.npz')
            # -- Training --
            # todo: ensure that all algorithms have the following method!
            alg.fit(train_loader.dataset.iteration_matrix)

            # -- Validation --
            metrics_values = evaluate_recommender_algorithm(alg, val_loader, conf['seed'] + 1)
            tune.report(**metrics_values)

            # -- Save --
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
    log_callback = WandbLoggerCallback(project=PROJECT_NAME, log_config=True, api_key_file=WANDB_API_KEY_PATH,
                                       reinit=True, force=True, job_type='train/val', tags=run_name.split('_'))

    keep_callback = KeepOnlyTopTrials(metric_name, n_tops=3)

    # Stopper
    stopper = CombinedStopper(
        NoImprovementsStopper(metric_name, max_patience=10),
        TrialPlateauStopper(metric_name, std=1e-3, num_results=5, grace_period=10)
    )

    # Other experiment's settings
    conf['experiment_settings'] = kwargs

    tune.register_trainable(run_name, tune_training)
    analysis = tune.run(
        run_name,
        config=conf,
        name=generate_id(prefix=run_name),
        resources_per_trial={'gpu': kwargs['n_gpus']},
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=kwargs['n_samples'],
        callbacks=[log_callback, keep_callback],
        metric=metric_name,
        stop=stopper,
        max_concurrent_trials=kwargs['n_concurrent'],
        mode='max',
        fail_fast=True,
    )

    best_value, best_checkpoint, best_config = keep_callback.get_best_trial()
    # best_trial = analysis.get_best_trial(metric_name, 'max', scope='all')
    # best_config = best_trial.config
    # best_checkpoint = analysis.get_best_checkpoint(best_trial, metric_name, 'max')

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
    if isinstance(alg, SGDBasedRecommenderAlgorithm):
        best_checkpoint = os.path.join(best_checkpoint, 'best_model.pth')
    else:
        best_checkpoint = os.path.join(best_checkpoint, 'best_model.npz')
    alg.load_model_from_path(best_checkpoint)
    if best_config['alg'] in [RecAlgorithmsEnum.pop, RecAlgorithmsEnum.rand]:
        metrics_values = evaluate_naive_algorithm(alg, test_loader, best_config['seed'] + 2)
    else:
        metrics_values = evaluate_recommender_algorithm(alg, test_loader, best_config['seed'] + 2)

    wandb.log(metrics_values)

    wandb.finish()

    return metrics_values


def start_hyper(alg: RecAlgorithmsEnum, dataset: RecDatasetsEnum, seed: int = SINGLE_SEED, **kwargs) -> dict:
    print('Starting Hyperparameter Optimization')
    print(f'Dataset is {dataset.name} - Seed is {seed}')
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
    best_config, best_checkpoint = None, ''
    if alg not in [RecAlgorithmsEnum.pop, RecAlgorithmsEnum.rand]:
        print('Start Train/Val')
        best_config, best_checkpoint = run_train_val(conf, run_name, **kwargs)
    else:
        best_config = conf

    print('Start Test')
    metric_values = run_test(run_name, best_config, best_checkpoint)

    print('End')
    return metric_values


def start_multiple_hyper(alg: RecAlgorithmsEnum, dataset: RecDatasetsEnum, **kwargs):
    print('Starting Multi-Hyperparameter Optimization')
    print(f'Dataset is {dataset.name} - Seeds are {SEED_LIST}')

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

    run_name = f'{alg.name}_{dataset.name}'
    wandb.login(key=wandb_api_key)
    wandb.init(project=PROJECT_NAME, group='aggr_results', name=run_name, force=True, job_type='test',
               tags=run_name.split('_'))

    wandb.log(aggr_results_dict)
    wandb.finish()


def start_multi_dataset(alg: RecAlgorithmsEnum, **kwargs):
    print('Starting Multi-dataset Multi-Hyperparameter Optimization')
    for dataset in RecDatasetsEnum:
        start_multiple_hyper(alg, dataset, **kwargs)
