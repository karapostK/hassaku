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
from algorithms.neural_alg import SGDMatrixFactorization
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
            prefetch_factor=5
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
            num_workers=2
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
            num_workers=2
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
    elif alg in [RecAlgorithmsEnum.sgdmf]:
        # Need to build the trainer
        exp_conf = ExperimentConfig(n_epochs=conf['n_epochs'],
                                    rec_loss=conf['rec_loss'].value(),
                                    lr=conf['optim_param']['lr'],
                                    wd=conf['optim_param']['wd'],
                                    optim_type=conf['optim_param']['optim'],
                                    )
        alg = SGDMatrixFactorization(dataloader.dataset.n_users, dataloader.dataset.n_items, conf['embedding_dim'])
        return alg, exp_conf


def tune_training(conf: dict, checkpoint_dir=None):
    """
    Function executed by ray tune.
    """
    train_loader = load_data(conf, 'train')
    val_loader = load_data(conf, 'val')

    reproducible(conf['seed'])

    alg = build_algorithm(conf['alg'], conf, train_loader)

    if conf['alg'] in [RecAlgorithmsEnum.svd, RecAlgorithmsEnum.uknn, RecAlgorithmsEnum.iknn, RecAlgorithmsEnum.slim]:
        # -- Training --
        alg.fit(train_loader.dataset.iteration_matrix)
        # -- Validation --
        metrics_values = evaluate_recommender_algorithm(alg, val_loader, conf['seed'])
        tune.report(**metrics_values)

        # -- Save --
        # We only keep the top-3 best performing models. Each trial saves the model if it is currently one of the best three.
        # When a trial becomes one of the top-3, it deletes the saved model of the trial with the current smallest metric value in the top-3.
        run_metric = metrics_values[OPTIMIZING_METRIC]
        to_save = False
        with tune.checkpoint_dir(0) as checkpoint_dir:
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

                        to_save = True

                        # Delete previous trial
                        old_path = top_paths[argmin]
                        if os.path.isdir(old_path):
                            shutil.rmtree(old_path)

                        top_values[argmin] = run_metric
                        top_paths[argmin] = checkpoint_dir

                        with open(sync_file_path, 'w') as out_file:
                            json.dump({'paths': top_paths, 'values': top_values}, out_file)

            # Save
            if to_save:
                if conf['alg'] == RecAlgorithmsEnum.svd:
                    np.savez(os.path.join(checkpoint_dir, 'best_model.npz'), users_factors=alg.users_factors,
                             items_factors=alg.items_factors)
                elif conf['alg'] in [RecAlgorithmsEnum.uknn, RecAlgorithmsEnum.iknn]:
                    np.savez(os.path.join(checkpoint_dir, 'best_model.npz'),
                             pred_mtx=alg.pred_mtx)
                elif conf['alg'] == RecAlgorithmsEnum.slim:
                    np.savez(os.path.join(checkpoint_dir, 'best_model.npz'),
                             pred_mtx=alg.pred_mtx)

        return metrics_values
    elif conf['alg'] in [RecAlgorithmsEnum.sgdmf]:
        # Training
        alg, expconf = alg
        trainer = Trainer(alg, train_loader, val_loader, expconf)
        trained_alg = trainer.fit()
        # No need of validation since it's carried out internally by the trainer.


def run_train_val(conf: dict, run_name: str):
    best_config = None
    best_checkpoint = None

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
        stopper = TrialPlateauStopper(metric_name, std=1e-3, num_results=5, grace_period=10)

        tune.register_trainable(run_name, tune_training)
        analysis = tune.run(
            run_name,
            config=conf,
            name=generate_id(prefix=run_name),
            resources_per_trial={'gpu': 0, 'cpu': 1},
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=NUM_SAMPLES,
            callbacks=[callback],
            metric=metric_name,
            stop=stopper,
            max_concurrent_trials=5,
            mode='max'
        )
        best_trial = analysis.get_best_trial(metric_name, 'max', scope='all')
        best_config = best_trial.config
        best_checkpoint = analysis.get_best_checkpoint(best_trial, metric_name, 'max')
        # todo: implement deletion of the other runs (maybe)

    return best_config, best_checkpoint


def run_test(run_name: str, best_config: dict, best_checkpoint=None):
    test_loader = load_data(best_config, 'test')

    with open(WANDB_API_KEY_PATH) as wandb_file:
        wandb_api_key = wandb_file.read()

    wandb.login(key=wandb_api_key)
    wandb.init(project=PROJECT_NAME, group='test', config=best_config, name=run_name, force=True,
               job_type='test', tags=run_name.split('_'))

    # ---- Test ---- #
    alg = build_algorithm(best_config['alg'], best_config, test_loader)

    if best_config['alg'] in [RecAlgorithmsEnum.random, RecAlgorithmsEnum.popular]:
        metrics_values = evaluate_recommender_algorithm(alg, test_loader, best_config['seed'])
        wandb.log(metrics_values)
    elif best_config['alg'] in [RecAlgorithmsEnum.svd]:
        best_checkpoint = os.path.join(best_checkpoint, 'best_model.npz')
        with np.load(best_checkpoint) as array_dict:
            alg.users_factors = array_dict['users_factors']
            alg.items_factors = array_dict['items_factors']
        metrics_values = evaluate_recommender_algorithm(alg, test_loader, best_config['seed'])
        wandb.log(metrics_values)
    elif best_config['alg'] in [RecAlgorithmsEnum.uknn, RecAlgorithmsEnum.iknn]:
        best_checkpoint = os.path.join(best_checkpoint, 'best_model.npz')
        with np.load(best_checkpoint) as array_dict:
            alg.pred_mtx = array_dict['pred_mtx']
        metrics_values = evaluate_recommender_algorithm(alg, test_loader, best_config['seed'])
        wandb.log(metrics_values)
    elif best_config['alg'] in [RecAlgorithmsEnum.slim]:
        with np.load(best_checkpoint) as array_dict:
            alg.pred_mtx = array_dict['pred_mtx']
        metrics_values = evaluate_recommender_algorithm(alg, test_loader, best_config['seed'])
        wandb.log(metrics_values)
    elif best_config['alg'] in [RecAlgorithmsEnum.sgdmf]:
        alg, expconf = alg
        # TODO: TO CONTINUE
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
