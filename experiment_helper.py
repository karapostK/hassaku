import json
import os
import shutil

import numpy as np
from filelock import FileLock
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.suggest.hyperopt import HyperOptSearch

import wandb
from data.dataset import get_recdataset_dataloader
from hyper_params import alg_param
from utilities.consts import NEG_VAL, SINGLE_SEED, PROJECT_NAME, WANDB_API_KEY_PATH, DATA_PATH, OPTIMIZING_METRIC, \
    NUM_SAMPLES
from utilities.enums import RecAlgorithmsEnum
from utilities.eval import evaluate_recommender_algorithm
from utilities.similarities import SimilarityFunctionEnum
from utilities.trainer import ExperimentConfig, Trainer
from utilities.utils import generate_id, reproducible


def load_data(conf: dict, split_set: str):
    if split_set == 'train':
        train_loader = get_recdataset_dataloader(
            'inter',
            data_path=conf['data_path'],
            split_set='train',
            n_neg=conf['neg_train'] if "neg_train" in conf else 4,
            neg_strategy=conf['train_neg_stratey'] if 'train_neg_stratey' in conf else 'uniform',
            batch_size=conf['batch_size'] if 'batch_size' in conf else 64,
            shuffle=True,
            num_workers=2,
            #prefetch_factor=5
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


def build_algorithm(alg: RecAlgorithmsEnum, conf: dict, dataloader):
    if alg == RecAlgorithmsEnum.random:
        return RecAlgorithmsEnum.random.value()
    elif alg == RecAlgorithmsEnum.popular:
        return RecAlgorithmsEnum.popular.value(dataloader.dataset.pop_distribution)
    elif alg == RecAlgorithmsEnum.svd:
        return RecAlgorithmsEnum.svd.value(conf['n_factors'])
    elif alg in [RecAlgorithmsEnum.uknn, RecAlgorithmsEnum.iknn]:
        sim_func_params = conf['sim_func_params']
        k = conf['k']
        sim_func = SimilarityFunctionEnum[sim_func_params['sim_func_name']]
        alpha = sim_func_params['alpha'] if 'alpha' in sim_func_params else None
        beta = sim_func_params['beta'] if 'beta' in sim_func_params else None
        if alg == RecAlgorithmsEnum.uknn:
            return RecAlgorithmsEnum.uknn.value(sim_func, k, alpha=alpha, beta=beta)
        else:
            return RecAlgorithmsEnum.iknn.value(sim_func, k, alpha=alpha, beta=beta)
    elif alg == RecAlgorithmsEnum.slim:
        return RecAlgorithmsEnum.slim.value(conf['alpha'], conf['l1_ratio'], conf['max_iter'])
    elif alg == RecAlgorithmsEnum.als:
        return RecAlgorithmsEnum.als.value(conf['alpha'], conf['factors'], conf['regularization'], conf['n_iterations'])
    elif alg == RecAlgorithmsEnum.rbmf:
        return RecAlgorithmsEnum.rbmf.value(conf['n_representatives'], conf['lam'])
    elif alg in [RecAlgorithmsEnum.sgdmf]:
        # Need to build the config for the trainer

        if dataloader.dataset.split_set == 'train':
            exp_conf = ExperimentConfig(n_epochs=conf['n_epochs'],
                                        rec_loss=conf['rec_loss'].value(),
                                        lr=conf['optim_param']['lr'],
                                        wd=conf['optim_param']['wd'],
                                        optim_type=conf['optim_param']['optim'],
                                        )
            alg = RecAlgorithmsEnum.sgdmf.value(dataloader.dataset.n_users, dataloader.dataset.n_items,
                                                conf['embedding_dim'])
            return alg, exp_conf
        else:
            alg = RecAlgorithmsEnum.sgdmf.value(dataloader.dataset.n_users, dataloader.dataset.n_items,
                                                conf['embedding_dim'])
            return alg


def check_whether_to_save(run_metric: float, checkpoint_dir: str) -> bool:
    """
    This function manages the saving of the found models by saving only the top-3 best models for a specific algorithm.
    This avoids saving all NUM_SAMPLES configurations of models.
    If the trial becomes part of the top-3, it deletes the model with the current smallest metric value among the top-3. Return to_save=True
    If the trial does not become part of the top-3, do nothing and return to_save=False
    :param run_metric: the metric of the current trial which will be used to check whether we save the model or not
    :param checkpoint_dir: the path pointing at the model
    :return: boolean value indicating whether to save the current model
    """

    with FileLock('../file.lock'):
        sync_file_path = '../sync_data.json'
        # Create file if not there the first time
        if not os.path.isfile(sync_file_path):
            with open(sync_file_path, 'w') as out_file:
                json.dump({'paths': [''] * 3, 'values': [-np.inf] * 3}, out_file)
        # Read the file used for synchronization
        with open(sync_file_path, 'r') as in_file:
            sync_data = json.load(in_file)
            top_paths = sync_data['paths']
            top_values = sync_data['values']

            # Compare the current trial with the trial that has the minimum metric value within the top-3
            argmin = np.argmin(top_values)
            if top_values[argmin] < run_metric:
                # Save
                print('--- Run saved ---', top_values, run_metric)

                # Delete previous trial
                old_path = top_paths[argmin]
                if os.path.isdir(old_path):
                    shutil.rmtree(old_path)

                top_values[argmin] = run_metric
                top_paths[argmin] = checkpoint_dir

                with open(sync_file_path, 'w') as out_file:
                    json.dump({'paths': top_paths, 'values': top_values}, out_file)
                return True
            else:
                return False


def tune_training(conf: dict, checkpoint_dir=None):
    """
    Function executed by ray tune.
    """
    train_loader = load_data(conf, 'train')
    val_loader = load_data(conf, 'val')

    reproducible(conf['seed'])

    alg = build_algorithm(conf['alg'], conf, train_loader)

    if conf['alg'] in [RecAlgorithmsEnum.svd, RecAlgorithmsEnum.uknn, RecAlgorithmsEnum.iknn, RecAlgorithmsEnum.slim,
                       RecAlgorithmsEnum.als, RecAlgorithmsEnum.rbmf]:
        # -- Training --
        alg.fit(train_loader.dataset.iteration_matrix)

        # -- Validation --
        metrics_values = evaluate_recommender_algorithm(alg, val_loader, conf['seed'])
        tune.report(**metrics_values)

        # -- Save --
        run_metric = metrics_values[OPTIMIZING_METRIC]

        with tune.checkpoint_dir(0) as checkpoint_dir:

            checkpoint_dir = os.path.join(checkpoint_dir, 'best_model.npz')
            to_save = check_whether_to_save(run_metric, checkpoint_dir)

            # Save
            if to_save:
                alg.save_model_to_path(checkpoint_dir)

    elif conf['alg'] in [RecAlgorithmsEnum.sgdmf]:
        # Training
        alg, exp_conf = alg

        with tune.checkpoint_dir(0) as checkpoint_dir:
            exp_conf.best_model_path = os.path.join(checkpoint_dir, 'best_model.npz')
            trainer = Trainer(alg, train_loader, val_loader, exp_conf)
            trainer.fit()
            to_save = check_whether_to_save(trainer.best_value, os.path.dirname(trainer.best_model_path))

            if not to_save:
                # Delete self since it shouldn't have been saved
                if os.path.isdir(os.path.dirname(trainer.best_model_path)):
                    shutil.rmtree(os.path.dirname(trainer.best_model_path))


def run_train_val(conf: dict, run_name: str):
    best_config = None
    best_checkpoint = ''

    print(conf)

    with open(WANDB_API_KEY_PATH) as wandb_file:
        wandb_api_key = wandb_file.read()

    # ---- Train and Val ---- #
    if conf['alg'] in [RecAlgorithmsEnum.random, RecAlgorithmsEnum.popular]:
        # No training and hyperparameter selection needed
        # Just evaluate the algorithm on validation data
        best_config = conf
        val_loader = load_data(best_config, 'val')
        alg = build_algorithm(best_config['alg'], best_config, val_loader)

        metrics_values = evaluate_recommender_algorithm(alg, val_loader, conf['seed'])
        wandb.login(key=wandb_api_key)
        wandb.init(project=PROJECT_NAME, group=run_name, config=best_config, name=run_name, force=True,
                   job_type='train/val', tags=run_name.split('_'))
        wandb.log(metrics_values)
        wandb.finish()
    else:
        # Hyperparameter Optimization
        metric_name = OPTIMIZING_METRIC

        # Search Algorithm
        search_alg = HyperOptSearch(random_state_seed=conf['seed'])

        if os.path.basename(conf['data_path']) == 'lfm2b-1m':
            scheduler = ASHAScheduler(grace_period=4)
        else:
            scheduler = None

        # Logger
        callback = WandbLoggerCallback(project=PROJECT_NAME, log_config=True, api_key=wandb_api_key,
                                       reinit=True, force=True, job_type='train/val', tags=run_name.split('_'))

        # Stopper
        # stopper = TrialPlateauStopper(metric_name, std=1e-3, num_results=5, grace_period=10)

        tune.register_trainable(run_name, tune_training)
        analysis = tune.run(
            run_name,
            config=conf,
            name=generate_id(prefix=run_name),
            #resources_per_trial={'gpu': 0, },#'cpu': 1}, # TODO mazbe remove cpu=1
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=NUM_SAMPLES,
            callbacks=[callback],
            metric=metric_name,
            # stop=stopper,
            max_concurrent_trials=3,
            mode='max'
        )
        best_trial = analysis.get_best_trial(metric_name, 'max', scope='all')
        best_config = best_trial.config
        best_checkpoint = analysis.get_best_checkpoint(best_trial, metric_name, 'max')

    return best_config, best_checkpoint


def run_test(run_name: str, best_config: dict, best_checkpoint=''):
    test_loader = load_data(best_config, 'test')

    with open(WANDB_API_KEY_PATH) as wandb_file:
        wandb_api_key = wandb_file.read()

    wandb.login(key=wandb_api_key)
    wandb.init(project=PROJECT_NAME, group='test', config=best_config, name=run_name, force=True,
               job_type='test', tags=run_name.split('_'))

    # ---- Test ---- #
    alg = build_algorithm(best_config['alg'], best_config, test_loader)

    best_checkpoint = os.path.join(best_checkpoint, 'best_model.npz')
    alg.load_model_from_path(best_checkpoint)
    metrics_values = evaluate_recommender_algorithm(alg, test_loader, best_config['seed'])
    wandb.log(metrics_values)

    wandb.finish()

    return metrics_values


def start_hyper(alg: RecAlgorithmsEnum, dataset: str, seed: int = SINGLE_SEED):
    print('Starting Hyperparameter Optimization')
    print(f'Seed is {seed}')

    # ---- Algorithm's parameters and hyperparameters ---- #
    conf = alg_param[alg]
    conf['alg'] = alg

    # Dataset
    conf['data_path'] = os.path.join(os.getcwd(), DATA_PATH, dataset)

    # Seed
    conf['seed'] = seed

    # Hostname
    host_name = os.uname()[1][:2]

    run_name = f'{alg.name}_{dataset}_{host_name}_{seed}'

    # ---- Train/Validation ---- #
    print('Start Train/Val')
    best_config, best_checkpoint = run_train_val(conf, run_name)

    print('Start Test')
    metric_values = run_test(run_name, best_config, best_checkpoint)

    print('End')
    return metric_values
