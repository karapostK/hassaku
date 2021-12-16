import argparse
import os

import wandb
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.suggest.hyperopt import HyperOptSearch
from rec_sys.protorec_dataset import get_protorecdataset_dataloader
from rec_sys.tester import Tester
from rec_sys.trainer import Trainer

from data.dataset import get_recdataset_dataloader, get_userrecdataset_dataloader
from hyper_params import alg_param
from utilities.consts import NEG_VAL, OPTIMIZING_METRIC, SINGLE_SEED, NUM_SAMPLES, \
    PROJECT_NAME, WANDB_API_KEY_PATH, DATA_PATH
from utilities.enums import RecAlgorithmsEnum
from utilities.utils import reproducible, generate_id


def load_data(conf: argparse.Namespace, split_set: str, dataset_type: str = 'inter'):
    assert split_set in ['train', 'val', 'test'], f'Split set ({split_set}) invalid'
    assert dataset_type in ['inter', 'user'], f'Type of Dataset ({dataset_type}) not defined'

    if dataset_type == 'inter':
        if split_set == 'train':
            train_loader = get_recdataset_dataloader(
                dataset_type=dataset_type,
                data_path=conf.data_path,
                split_set='train',
                n_neg=conf.neg_train,
                neg_sampling_strategy=conf.train_neg_sampling_strategy,
                batch_size=conf.batch_size,
                shuffle=True,
                num_workers=2,
                prefetch_factor=5
            )
        elif split_set == 'val':
            val_loader = get_recdataset_dataloader(
                dataset_type=dataset_type,
                data_path=conf.data_path,
                split_set='val',
                n_neg=NEG_VAL,
                neg_sampling_strategy=conf.eval_neg_sampling_strategy,
                batch_size=conf.val_batch_size,
                num_workers=2
            )
        else:
            test_loader = get_recdataset_dataloader(
                dataset_type=dataset_type,
                data_path=conf.data_path,
                split_set='test',
                n_neg=NEG_VAL,
                neg_sampling_strategy=conf.eval_neg_sampling_strategy,
                batch_size=conf.val_batch_size,
                num_workers=2
            )
    else:
        if split_set == 'train':
            train_loader = get_recdataset_dataloader(
                dataset_type=dataset_type,
                data_path=conf.data_path,
                split_set='train',
                n_neg=conf.neg_train if 'neg_train' in conf else None,
                neg_sampling_strategy=conf.train_neg_sampling_strategy,
                pos_strategy='n_pos',
                neg_strategy='ratio',
                neg_pos_ratio=3,
                n_pos=100,
                batch_size=conf.batch_size,
                shuffle=True,
                num_workers=2,
                prefetch_factor=5
            )
            elif split_set == 'val':
            val_loader = get_protorecdataset_dataloader(
                dataset_type=dataset_type,
                data_path=conf.data_path,
                split_set='val',
                n_neg=NEG_VAL,
                neg_strategy=conf.eval_neg_strategy,
                batch_size=conf.val_batch_size,
                num_workers=2
            )
        else:
            test_loader = get_protorecdataset_dataloader(
                dataset_type=dataset_type,
                data_path=conf.data_path,
                split_set='test',
                n_neg=NEG_VAL,
                neg_strategy=conf.eval_neg_strategy,
                batch_size=conf.val_batch_size,
                num_workers=2
            )
        dataloader_getter = get_userrecdataset_dataloader

    if split_set == 'train':
        train_loader = get_recdataset_dataloader(
            data_path=conf.data_path,
            split_set='train',
            n_neg=conf.neg_train,
            neg_strategy=conf.train_neg_strategy,
            batch_size=conf.batch_size,
            shuffle=True,
            num_workers=2,
            prefetch_factor=5
        )
    elif split_set == 'val':
        val_loader = get_protorecdataset_dataloader(
            data_path=conf.data_path,
            split_set='val',
            n_neg=NEG_VAL,
            neg_strategy=conf.eval_neg_strategy,
            batch_size=conf.val_batch_size,
            num_workers=2
        )
    else:

        test_loader = get_protorecdataset_dataloader(
            data_path=conf.data_path,
            split_set='test',
            n_neg=NEG_VAL,
            neg_strategy=conf.eval_neg_strategy,
            batch_size=conf.val_batch_size,
            num_workers=2
        )

        return {'test_loader': test_loader}


def start_training(config, checkpoint_dir=None):
    config = argparse.Namespace(**config)
    print(config)

    # ---- Dataloader ---- #

    data_loaders_dict = load_data(config)

    reproducible(config.seed)

    trainer = Trainer(data_loaders_dict['train_loader'], data_loaders_dict['val_loader'], config)

    trainer.run()

    wandb.finish()


def start_testing(config, model_load_path: str):
    config = argparse.Namespace(**config)
    print(config)

    data_loaders_dict = load_data(config, is_train=False)

    reproducible(config.seed)

    tester = Tester(data_loaders_dict['test_loader'], config, model_load_path)

    metric_values = tester.test()
    return metric_values


def start_hyper(alg: RecAlgorithmsEnum, dataset: str, seed: int = SINGLE_SEED):
    print('Starting Hyperparameter Optimization')
    print(f'Seed is {seed}')

    # ---- Preparing parameters for tune.run ---- #
    metric_name = '_metric/' + OPTIMIZING_METRIC
    # Search Algorithm
    search_alg = HyperOptSearch(random_state_seed=seed)

    # Scheduler
    if dataset == 'lfm2b-1m':
        scheduler = ASHAScheduler(grace_period=4)
    else:
        scheduler = None

    # Logger
    with open(WANDB_API_KEY_PATH) as wandb_file:
        wandb_api_key = wandb_file.read()
    callback = WandbLoggerCallback(project=PROJECT_NAME, log_config=True, api_key=wandb_api_key,
                                   reinit=True, force=True, job_type='train/val', tags=[alg.name, str(seed), dataset])

    # Stopper
    stopper = TrialPlateauStopper(metric_name, std=1e-3, num_results=5, grace_period=10)

    # ---- Algorithm's parameters and hyperparameters ---- #
    conf = alg_param[alg]
    conf['alg'] = alg
    # Hostname
    host_name = os.uname()[1][:2]

    # Dataset
    conf['data_path'] = os.path.join(DATA_PATH, dataset)

    # Seed
    conf['seed'] = seed

    group_name = f'{alg.name}_{dataset}_{host_name}_{seed}'
    tune.register_trainable(group_name, start_training)
    analysis = tune.run(
        group_name,
        config=conf,
        name=generate_id(prefix=group_name),
        resources_per_trial={'gpu': 0.2, 'cpu': 1},
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=1 if alg in [RecAlgorithmsEnum.random, RecAlgorithmsEnum.popular] else NUM_SAMPLES,
        callbacks=[callback],
        metric=metric_name,
        mode='max',
        stop=stopper
    )

    # ---- Get the best trial ---- #
    best_trial = analysis.get_best_trial(metric_name, 'max',
                                         scope='all')  # take the trial that got the max over all epochs
    best_trial_config = best_trial.config
    best_trial_checkpoint = os.path.join(analysis.get_best_checkpoint(best_trial, metric_name, 'max'), 'best_model.pth')

    # ---- Test the best trial ---- #
    wandb.login(key=wandb_api_key)
    wandb.init(project=PROJECT_NAME, group='test_results', config=best_trial_config, name=group_name, force=True,
               job_type='test', tags=[alg.name, str(seed), dataset])
    metric_values = start_testing(best_trial_config, best_trial_checkpoint)
    wandb.finish()
    return metric_values

# def start_multiple_hyper(model: str, dataset: str, seed_list: List = SEED_LIST):
#     print('Starting Multi-Hyperparameter Optimization')
#     print('seed_list is ', seed_list)
#     metric_values_list = []
#     mean_values = dict()
#
#     for seed in seed_list:
#         metric_values_list.append(start_hyper(conf, model, dataset, seed))
#
#     for key in metric_values_list[0].keys():
#         _sum = 0
#         for metric_values in metric_values_list:
#             _sum += metric_values[key]
#         _mean = _sum / len(metric_values_list)
#
#         mean_values[key] = _mean
#
#     group_name = f'{model}_{dataset}'
#     with open('./wandb_api_key') as wandb_file:
#         wandb_api_key = wandb_file.read()
#     wandb.login(key=wandb_api_key)
#     wandb.init(project=PROJECT_NAME, group='aggr_results', name=group_name, force=True, job_type='test',
#                tags=[model, dataset])
#     wandb.log(mean_values)
#     wandb.finish()
