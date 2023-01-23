import os.path

import yaml

from algorithms.algorithms_utils import AlgorithmsEnum
from algorithms.base_classes import SGDBasedRecommenderAlgorithm
from data.data_utils import DatasetsEnum
from train.rec_losses import RecommenderSystemLossesEnum
from utilities.utils import generate_id

DEF_NEG_TRAIN = 4
DEF_NEG_STRATEGY = 'uniform'
DEF_TRAIN_BATCH_SIZE = 64
DEF_EVAL_BATCH_SIZE = 64
DEF_NUM_WORKERS = 2
DEF_SEED = 64
DEF_N_EPOCHS = 50
DEF_USE_WANDB = True
DEF_MODEL_SAVE_PATH = './saved_models'
DEF_LEARNING_RATE = 1e-3
DEF_WEIGHT_DECAY = 0
DEF_OPTIMIZER = 'adam'
DEF_REC_LOSS = 'bce'
DEF_LOSS_AGGREGATOR = 'mean'
DEF_DEVICE = 'cpu'
DEF_OPTIMIZING_METRIC = 'ndcg@10'
DEF_MAX_PATIENCE = DEF_N_EPOCHS - 1


def parse_yaml(conf_path: str) -> dict:
    assert os.path.isfile(conf_path), f'Configuration File {conf_path} not found!'

    with open(conf_path, 'r') as conf_file:
        conf = yaml.safe_load(conf_file)
    print(' --- Configuration Loaded ---')
    return conf


def save_yaml(conf_path: str, conf: dict):
    conf_path = os.path.join(conf_path, 'conf.yml')

    with open(conf_path, 'w') as conf_file:
        yaml.dump(conf, conf_file)
    print(' --- Configuration Saved ---')


def parse_conf(conf: dict, alg: AlgorithmsEnum, dataset: DatasetsEnum) -> dict:
    """
    It sets basic parameters of the configurations and provides default parameters
    """
    assert 'data_path' in conf, "Data path is missing from the configuration file"

    conf['alg'] = alg.name
    conf['time_run'] = generate_id()
    conf['dataset'] = dataset.name
    conf['data_path'] = conf['data_path']
    conf['dataset_path'] = os.path.join(conf['data_path'], conf['dataset'])

    # Adding default parameters
    added_parameters_list = []

    if 'model_save_path' not in conf:
        conf['model_save_path'] = DEF_MODEL_SAVE_PATH
        added_parameters_list.append(f"model_save_path={conf['model_save_path']}")

    conf['model_path'] = os.path.join(conf['model_save_path'], "{}-{}".format(alg.name, dataset.name), conf['time_run'])
    os.makedirs(conf['model_path'], exist_ok=True)

    if 'running_settings' not in conf:
        conf['running_settings'] = dict()

    if 'eval_batch_size' not in conf:
        conf['eval_batch_size'] = DEF_EVAL_BATCH_SIZE
        added_parameters_list.append(f"eval_batch_size={conf['eval_batch_size']}")

    if 'seed' not in conf['running_settings']:
        conf['running_settings']['seed'] = DEF_SEED
        added_parameters_list.append(f"seed={conf['running_settings']['seed']}")

    if 'use_wandb' not in conf['running_settings']:
        conf['running_settings']['use_wandb'] = DEF_USE_WANDB
        added_parameters_list.append(f"use_wandb={conf['running_settings']['use_wandb']}")

    if issubclass(alg.value, SGDBasedRecommenderAlgorithm):
        if 'neg_train' not in conf:
            conf['neg_train'] = DEF_NEG_TRAIN
            added_parameters_list.append(f"neg_train={conf['neg_train']}")

        if 'train_neg_strategy' not in conf:
            conf['train_neg_strategy'] = DEF_NEG_STRATEGY
            added_parameters_list.append(f"train_neg_strategy={conf['train_neg_strategy']}")

        if 'train_batch_size' not in conf:
            conf['train_batch_size'] = DEF_TRAIN_BATCH_SIZE
            added_parameters_list.append(f"train_batch_size={conf['train_batch_size']}")

        if 'n_epochs' not in conf:
            conf['n_epochs'] = DEF_N_EPOCHS
            added_parameters_list.append(f"n_epochs={conf['n_epochs']}")
        else:
            assert conf['n_epochs'] > 0, f"Number of epochs ({conf['n_epochs']}) should be positive"

        if 'lr' not in conf:
            conf['lr'] = DEF_LEARNING_RATE
            added_parameters_list.append(f"lr={conf['lr']}")
        else:
            assert conf['lr'] > 0, f"Learning rate ({conf['lr']}) should be positive"

        if 'wd' not in conf:
            conf['wd'] = DEF_WEIGHT_DECAY
            added_parameters_list.append(f"wd={conf['wd']}")
        else:
            assert conf['wd'] > 0, f"Weight Decay ({conf['wd']}) should be positive"

        if 'optimizer' not in conf:
            conf['optimizer'] = DEF_OPTIMIZER
            added_parameters_list.append(f"optimizer={conf['optimizer']}")
        else:
            assert conf['optimizer'] in ['adam', 'adagrad'], f"Optimizer ({conf['optimizer']}) not implemented"

        if 'rec_loss' not in conf:
            conf['rec_loss'] = DEF_REC_LOSS
            added_parameters_list.append(f"rec_loss={conf['rec_loss']}")
        else:
            assert conf['rec_loss'] in [rec_loss.name for rec_loss in
                                        RecommenderSystemLossesEnum], f"Rec loss ({conf['rec_loss']}) not implemented"

        if 'loss_aggregator' not in conf:
            conf['loss_aggregator'] = DEF_LOSS_AGGREGATOR
            added_parameters_list.append(f"loss_aggregator={conf['loss_aggregator']}")
        else:
            assert conf['loss_aggregator'] in ['mean',
                                               'sum'], f"Loss aggregator ({conf['loss_aggregator']}) not implemented"

        if 'device' not in conf:
            conf['device'] = DEF_DEVICE
            added_parameters_list.append(f"device={conf['device']}")
        else:
            assert conf['device'] in ['cpu',
                                      'cuda'], f"Device ({conf['device']}) not available"

        if 'optimizing_metric' not in conf:
            conf['optimizing_metric'] = DEF_OPTIMIZING_METRIC
            added_parameters_list.append(f"optimizing_metric={conf['optimizing_metric']}")

        if 'max_patience' not in conf:
            conf['max_patience'] = DEF_MAX_PATIENCE
            added_parameters_list.append(f"max_patience={conf['max_patience']}")
        else:
            assert 0 < conf['max_patience'] < conf[
                'n_epochs'], f"Max patience {conf['max_patience']} should be between 0 and {conf['n_epochs']}"

        if 'n_workers' not in conf['running_settings']:
            conf['running_settings']['n_workers'] = DEF_NUM_WORKERS
            added_parameters_list.append(f"n_workers={conf['running_settings']['n_workers']}")

    print('Added these default parameters: ', ", ".join(added_parameters_list))
    print('For more detail, see conf/conf_parser.py')
    print('\n\n')

    return conf
