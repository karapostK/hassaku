import os
from collections import defaultdict

import numpy as np
import torch
import wandb
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import CombinedStopper, TrialPlateauStopper
from torch.utils.data import DataLoader

from algorithms.base_classes import SGDBasedRecommenderAlgorithm
from algorithms.naive_algs import PopularItems, RandomItems
from consts.consts import PROJECT_NAME, WANDB_API_KEY_PATH, SEED_LIST, DATA_PATH, OPTIMIZING_METRIC, SINGLE_SEED, \
    ENTITY_NAME, EVAL_BATCH_SIZE
from consts.enums import RecAlgorithmsEnum, RecDatasetsEnum
from data.dataset import TrainRecDataset, FullEvalDataset
from hyper_search.hyper_params import alg_param
from hyper_search.utils import KeepOnlyTopTrials, NoImprovementsStopper, HyperOptSearchMaxMetric
from train.trainer import ExperimentConfig, Trainer
from utilities.eval import evaluate_recommender_algorithm
from utilities.utils import generate_id, reproducible


def load_data(conf: dict, split_set: str, **kwargs):
    if split_set == 'train':
        return DataLoader(
            TrainRecDataset(
                data_path=conf['data_path'],
                n_neg=conf['neg_train'] if "neg_train" in conf else 4,
                neg_sampling_strategy=conf['train_neg_strategy'] if 'train_neg_strategy' in conf else 'uniform',
            ),
            batch_size=conf['batch_size'] if 'batch_size' in conf else 64,
            shuffle=True,
            num_workers=kwargs['n_workers'] if 'n_workers' in kwargs else 2
        )

    elif split_set == 'val':
        return DataLoader(
            FullEvalDataset(
                data_path=conf['data_path'],
                split_set='val',
            ),
            batch_size=conf['eval_batch_size']
        )

    elif split_set == 'test':
        return DataLoader(
            FullEvalDataset(
                data_path=conf['data_path'],
                split_set='test',
            ),
            batch_size=conf['eval_batch_size']
        )


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
            conf['best_model_path'] = checkpoint_file
            exp_conf = ExperimentConfig.build_from_conf(conf)

            # Validation happens within Trainer
            trainer = Trainer(alg, train_loader, val_loader, exp_conf)
            trainer.fit()

        else:
            checkpoint_file = os.path.join(checkpoint_dir, 'best_model.npz')
            # -- Training --
            # todo: ensure that all algorithms have the following method!
            alg.fit(train_loader.dataset.iteration_matrix)

            # -- Validation --
            metrics_values = evaluate_recommender_algorithm(alg, val_loader, conf['seed'])
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
    search_alg = HyperOptSearchMaxMetric(random_state_seed=conf['seed'])

    if os.path.basename(conf['data_path']) == 'lfm2b1m':
        scheduler = ASHAScheduler(grace_period=4)
    else:
        scheduler = None

    # Logger
    log_callback = WandbLoggerCallback(project=PROJECT_NAME, log_config=True, api_key_file=WANDB_API_KEY_PATH,
                                       reinit=True, force=True, job_type='train/val', tags=run_name.split('_'),
                                       entity=ENTITY_NAME)

    keep_callback = KeepOnlyTopTrials(metric_name, n_tops=3)

    # Stopper
    stopper = CombinedStopper(
        NoImprovementsStopper(metric_name, max_patience=10),
        TrialPlateauStopper(metric_name, std=1e-5, num_results=5, grace_period=10)
    )

    # Other experiment's settings
    experiment_name = generate_id(prefix=run_name)

    conf['device'] = 'cuda' if kwargs['n_gpus'] > 0 and torch.cuda.is_available() else 'cpu'
    conf['experiment_settings'] = kwargs
    conf['run_name'] = run_name
    conf['experiment_name'] = experiment_name

    tune.register_trainable(run_name, tune_training)
    tune.run(
        run_name,
        config=conf,
        name=experiment_name,
        resources_per_trial={'gpu': kwargs['n_gpus'], 'cpu': kwargs['n_cpus']},
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=kwargs['n_samples'],
        callbacks=[log_callback, keep_callback],
        metric=metric_name,
        stop=stopper,
        max_concurrent_trials=kwargs['n_concurrent'],
        mode='max',
        fail_fast='raise',
    )

    best_value, best_checkpoint, best_config = keep_callback.get_best_trial()

    print('Train and Val ended')
    print(f'Best configuration is: \n {best_config}')
    print(f'Best checkpoint is: \n {best_checkpoint}')

    # Logging info to file for easier post-processing
    keep_callback.log_bests(os.path.expanduser(os.path.join('~/ray_results', experiment_name)))

    return best_config, best_checkpoint


def run_test(run_name: str, best_config: dict, best_checkpoint=''):
    """
    Runs the test procedure.
    """
    test_loader = load_data(best_config, 'test')

    wandb.login()
    wandb.init(project=PROJECT_NAME, group='test', config=best_config, name=run_name, force=True,
               job_type='test', tags=run_name.split('_'), entity=ENTITY_NAME)

    # ---- Test ---- #
    if best_config['alg'].value == PopularItems or best_config['alg'].value == RandomItems:
        train_loader = load_data(best_config, 'train')
        alg = best_config['alg'].value.build_from_conf(best_config, train_loader.dataset)
    else:
        alg = best_config['alg'].value.build_from_conf(best_config, test_loader.dataset)
        if isinstance(alg, SGDBasedRecommenderAlgorithm):
            best_checkpoint = os.path.join(best_checkpoint, 'best_model.pth')
        else:
            best_checkpoint = os.path.join(best_checkpoint, 'best_model.npz')
        alg.load_model_from_path(best_checkpoint)

    metrics_values = evaluate_recommender_algorithm(alg, test_loader, best_config['seed'])

    wandb.log(metrics_values)

    wandb.finish()

    return metrics_values


def start_hyper(alg: RecAlgorithmsEnum, dataset: RecDatasetsEnum, seed: int = SINGLE_SEED, **kwargs) -> dict:
    print('Starting Hyperparameter Optimization')
    print(f'Dataset is {dataset.name} - Seed is {seed}')

    # ---- Algorithm's parameters and hyperparameters ---- #
    conf = alg_param[alg.name]
    conf['alg'] = alg

    # Dataset
    conf['data_path'] = os.path.join(os.getcwd(), DATA_PATH, dataset.name)

    # Seed
    conf['seed'] = seed

    run_name = f'{alg.name}_{dataset.name}_{seed}'

    # ---- Train/Validation ---- #

    if alg.value == PopularItems or alg.value == RandomItems:
        # Skipping training and validation for naive algorithms
        best_config, best_checkpoint = conf, ''
    else:
        print('Start Train/Val')
        best_config, best_checkpoint = run_train_val(conf, run_name, **kwargs)

    print('Start Test')
    metric_values = run_test(run_name, best_config, best_checkpoint)

    print('End')
    return metric_values


def start_multiple_hyper(alg: RecAlgorithmsEnum, dataset: RecDatasetsEnum, **kwargs):
    print('Starting Multi-Hyperparameter Optimization')
    print(f'Dataset is {dataset.name} - Seeds are {SEED_LIST}')

    run_name = f'{alg.name}_{dataset.name}'

    # Checking whether we already run the experiments
    wapi = wandb.Api()
    multi_dataset_runs = wapi.runs(f'{ENTITY_NAME}/{PROJECT_NAME}', filters={'group': 'aggr_results'})
    multi_dataset_runs_names = set([x.name for x in multi_dataset_runs])
    if run_name in multi_dataset_runs_names:
        print(f'\n\n\n'
              f'Dataset <<{dataset.name}>> skipped since results are already available'
              f'\n\n\n')
        return

    # Checking whether we already partially run the experiments
    multi_hyper_runs = wapi.runs(f'{ENTITY_NAME}/{PROJECT_NAME}', filters={'group': 'test'})
    multi_hyper_runs_names = set([x.name for x in multi_hyper_runs])

    # Accumulate the results in a dictionary: e.g. results_list['ndcg@10'] = [0.8,0.5,0.3]
    results_dict = defaultdict(list)

    # Carry out the experiment
    for seed in SEED_LIST:
        sub_run_name = f'{alg.name}_{dataset.name}_{seed}'
        if sub_run_name in multi_hyper_runs_names:
            print(f'\n\n\n'
                  f'Dataset <<{dataset.name}>> and seed <<{seed}>> skipped since results are already available'
                  f'\n\n\n')
            run = [x for x in multi_hyper_runs if x.name == sub_run_name][0]
            metric_values = {k: v for k, v in run.summary.items() if k[0] != '_'}
        else:
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

    wandb.login(key=wandb_api_key)
    wandb.init(project=PROJECT_NAME, group='aggr_results', name=run_name, force=True, job_type='test',
               tags=run_name.split('_'))

    wandb.log(aggr_results_dict)
    wandb.finish()


def start_multi_dataset(alg: RecAlgorithmsEnum, **kwargs):
    print('Starting Multi-dataset Multi-Hyperparameter Optimization')

    # Checking whether we already run the experiments
    wapi = wandb.Api()
    multi_dataset_runs = wapi.runs(f'{ENTITY_NAME}/{PROJECT_NAME}', filters={'group': 'aggr_results'})
    multi_dataset_runs_names = set([x.name for x in multi_dataset_runs])

    for dataset in RecDatasetsEnum:
        run_name = f'{alg.name}_{dataset.name}'
        if run_name in multi_dataset_runs_names:
            print(f'\n\n\n'
                  f'Dataset <<{dataset.name}>> skipped since results are already available'
                  f'\n\n\n')
        else:
            start_multiple_hyper(alg, dataset, **kwargs)
