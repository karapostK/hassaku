import typing

import wandb
from algorithms.algorithms_utils import AlgorithmsEnum
from algorithms.base_classes import SGDBasedRecommenderAlgorithm, SparseMatrixBasedRecommenderAlgorithm
from algorithms.naive_algs import PopularItems
from conf.conf_parser import parse_conf_file, parse_conf, save_yaml
from data.data_utils import DatasetsEnum, get_dataloader
from data.dataset import TrainRecDataset
from eval.eval import evaluate_recommender_algorithm
from train.trainer import Trainer
from utilities.utils import reproducible
from wandb_conf import PROJECT_NAME, ENTITY_NAME


def train_val_agent():
    # Initialization and gathering hyperparameters
    run = wandb.init(job_type='train/val')

    conf = wandb.config

    alg = AlgorithmsEnum[conf.alg]
    dataset = DatasetsEnum[conf.dataset]

    conf = parse_conf(conf, alg, dataset)

    # Updating wandb data
    run.tags += (alg.name, dataset.name)
    wandb.config.update(conf)

    print('Starting Train-Val')
    print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')

    reproducible(conf['running_settings']['seed'])

    if issubclass(alg.value, SGDBasedRecommenderAlgorithm):

        train_loader = get_dataloader(conf, 'train')
        val_loader = get_dataloader(conf, 'val')

        alg = alg.value.build_from_conf(conf, train_loader.dataset)

        # Validation happens within the Trainer
        trainer = Trainer(alg, train_loader, val_loader, conf)
        trainer.fit()

    elif issubclass(alg.value, SparseMatrixBasedRecommenderAlgorithm):
        train_dataset = TrainRecDataset(conf['dataset_path'])
        val_loader = get_dataloader(conf, 'val')

        alg = alg.value.build_from_conf(conf, train_dataset)
        # -- Training --
        alg.fit(train_dataset.sampling_matrix)
        # -- Validation --
        metrics_values = evaluate_recommender_algorithm(alg, val_loader)

        alg.save_model_to_path(conf['model_path'])
        wandb.log(metrics_values)
    elif alg in [AlgorithmsEnum.rand, AlgorithmsEnum.pop]:

        train_dataset = TrainRecDataset(conf['dataset_path'])
        val_loader = get_dataloader(conf, 'val')

        alg = alg.value.build_from_conf(conf, train_dataset)
        metrics_values = evaluate_recommender_algorithm(alg, val_loader)
        wandb.log(metrics_values)
    else:
        raise ValueError(f'Training for {alg.value} has been not implemented')
    save_yaml(conf['model_path'], conf)
    wandb.finish()


def run_test(alg: AlgorithmsEnum, dataset: DatasetsEnum, conf: typing.Union[str, dict]):
    print('Starting Test')
    print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')

    if isinstance(conf, str):
        conf = parse_conf_file(conf)

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
        wandb.log(metrics_values, step=0)
        wandb.finish()


train_val_agent()
