import typing

import wandb
from algorithms.algorithms_utils import AlgorithmsEnum
from algorithms.base_classes import SGDBasedRecommenderAlgorithm, SparseMatrixBasedRecommenderAlgorithm
from algorithms.naive_algs import PopularItems
from conf.conf_parser import parse_yaml, parse_conf, save_yaml
from data.data_utils import DatasetsEnum, get_dataloader
from data.dataset import TrainRecDataset
from eval.eval import evaluate_recommender_algorithm
from train.trainer import Trainer
from utilities.utils import reproducible
from wandb_conf import PROJECT_NAME, ENTITY_NAME


def run_train_val(alg: AlgorithmsEnum, dataset: DatasetsEnum, conf: typing.Union[str, dict]):
    print('Starting Train-Val')
    print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')

    if isinstance(conf, str):
        conf = parse_yaml(conf)
    conf = parse_conf(conf, alg, dataset)

    if conf['running_settings']['use_wandb']:
        wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME, config=conf, tags=[alg.name, dataset.name],
                   group=f'{alg.name} - {dataset.name} - train/val', name=conf['time_run'], job_type='train/val')

    reproducible(conf['running_settings']['seed'])

    if issubclass(alg.value, SGDBasedRecommenderAlgorithm):

        train_loader = get_dataloader(conf, 'train')
        val_loader = get_dataloader(conf, 'val')

        alg = alg.value.build_from_conf(conf, train_loader.dataset)

        # Validation happens within the Trainer
        trainer = Trainer(alg, train_loader, val_loader, conf)
        metrics_values = trainer.fit()
        save_yaml(conf['model_path'], conf)

    elif issubclass(alg.value, SparseMatrixBasedRecommenderAlgorithm):
        train_dataset = TrainRecDataset(conf['dataset_path'])
        val_loader = get_dataloader(conf, 'val')

        alg = alg.value.build_from_conf(conf, train_dataset)
        # -- Training --
        alg.fit(train_dataset.iteration_matrix)
        # -- Validation --
        metrics_values = evaluate_recommender_algorithm(alg, val_loader)

        alg.save_model_to_path(conf['model_path'])
        save_yaml(conf['model_path'], conf)

        if conf['running_settings']['use_wandb']:
            wandb.log(metrics_values)
    elif alg in [AlgorithmsEnum.rand, AlgorithmsEnum.pop]:

        train_dataset = TrainRecDataset(conf['dataset_path'])
        val_loader = get_dataloader(conf, 'val')

        alg = alg.value.build_from_conf(conf, train_dataset)
        metrics_values = evaluate_recommender_algorithm(alg, val_loader)
        if conf['running_settings']['use_wandb']:
            wandb.log(metrics_values)
    else:
        raise ValueError(f'Training for {alg.value} has been not implemented')

    return metrics_values, conf


def run_test(alg: AlgorithmsEnum, dataset: DatasetsEnum, conf: typing.Union[str, dict]):
    print('Starting Test')
    print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')

    if isinstance(conf, str):
        conf = parse_yaml(conf)

    if conf['running_settings']['use_wandb']:
        wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME, config=conf, tags=[alg.name, dataset.name],
                   group=f'{alg.name} - {dataset.name} - test', name=conf['time_run'], job_type='test', reinit=True)

    test_loader = get_dataloader(conf, 'test')

    if alg.value == PopularItems:
        # Popular Items requires the popularity distribution over the items learned over the training data
        alg = alg.value.build_from_conf(conf, TrainRecDataset(conf['dataset_path']))
    else:
        alg = alg.value.build_from_conf(conf, test_loader.dataset)

    alg.load_model_from_path(conf['model_path'])

    metrics_values = evaluate_recommender_algorithm(alg, test_loader)
    if conf['running_settings']['use_wandb']:
        wandb.log(metrics_values)


def run_train_val_test(alg: AlgorithmsEnum, dataset: DatasetsEnum, conf_path: str):
    print('Starting Train-Val-Test')
    print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')

    # ------ Run train and Val ------ #
    metrics_values, conf = run_train_val(alg, dataset, conf_path)
    # ------ Run test ------ #
    run_test(alg, dataset, conf)
