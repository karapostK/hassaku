import typing

import wandb
from algorithms.algorithms_utils import AlgorithmsEnum
from algorithms.base_classes import SGDBasedRecommenderAlgorithm, SparseMatrixBasedRecommenderAlgorithm
from algorithms.naive_algs import PopularItems
from algorithms.sgd_alg import ECF
from conf.conf_parser import parse_conf_file, parse_conf, save_yaml
from data.data_utils import DatasetsEnum, get_dataloader
from data.dataset import TrainRecDataset, ECFTrainRecDataset
from eval.eval import evaluate_recommender_algorithm, FullEvaluator
from train.rec_losses import RecommenderSystemLoss
from train.trainer import Trainer
from utilities.utils import reproducible
from wandb_conf import PROJECT_NAME, ENTITY_NAME


def run_train_val(alg: AlgorithmsEnum, dataset: DatasetsEnum, conf: typing.Union[str, dict]):
    print('Starting Train-Val')
    print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')

    if isinstance(conf, str):
        conf = parse_conf_file(conf)
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
        rec_loss = RecommenderSystemLoss.build_from_conf(conf, train_loader.dataset)
        trainer = Trainer(alg, train_loader, val_loader, rec_loss, conf)
        metrics_values = trainer.fit()
        save_yaml(conf['model_path'], conf)

    elif issubclass(alg.value, SparseMatrixBasedRecommenderAlgorithm):
        train_dataset = TrainRecDataset(conf['dataset_path'])
        val_loader = get_dataloader(conf, 'val')

        alg = alg.value.build_from_conf(conf, train_dataset)
        # -- Training --
        alg.fit(train_dataset.sampling_matrix)
        # -- Validation --
        evaluator = FullEvaluator(aggr_by_group=True, n_groups=val_loader.dataset.n_user_groups,
                                  user_to_user_group=val_loader.dataset.user_to_user_group)
        metrics_values = evaluate_recommender_algorithm(alg, val_loader, evaluator,
                                                        verbose=conf['running_settings']['batch_verbose'])

        alg.save_model_to_path(conf['model_path'])
        save_yaml(conf['model_path'], conf)

        if conf['running_settings']['use_wandb']:
            wandb.log(metrics_values)
    elif alg in [AlgorithmsEnum.rand, AlgorithmsEnum.pop]:

        train_dataset = TrainRecDataset(conf['dataset_path'])
        val_loader = get_dataloader(conf, 'val')

        alg = alg.value.build_from_conf(conf, train_dataset)
        evaluator = FullEvaluator(aggr_by_group=True, n_groups=val_loader.dataset.n_user_groups,
                                  user_to_user_group=val_loader.dataset.user_to_user_group)
        metrics_values = evaluate_recommender_algorithm(alg, val_loader, evaluator,
                                                        verbose=conf['running_settings']['batch_verbose'])

        save_yaml(conf['model_path'], conf)

        if conf['running_settings']['use_wandb']:
            wandb.log(metrics_values)
    else:
        raise ValueError(f'Training for {alg.value} has been not implemented')

    if conf['running_settings']['use_wandb']:
        wandb.finish()

    return metrics_values, conf


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
    elif alg.value == ECF:
        alg = alg.value.build_from_conf(conf, ECFTrainRecDataset(conf['dataset_path']))
    else:
        alg = alg.value.build_from_conf(conf, test_loader.dataset)

    alg.load_model_from_path(conf['model_path'])

    evaluator = FullEvaluator(aggr_by_group=True, n_groups=test_loader.dataset.n_user_groups,
                              user_to_user_group=test_loader.dataset.user_to_user_group)
    metrics_values = evaluate_recommender_algorithm(alg, test_loader, evaluator,
                                                    verbose=conf['running_settings']['batch_verbose'])
    if conf['running_settings']['use_wandb']:
        wandb.log(metrics_values, step=0)
        wandb.finish()


def run_train_val_test(alg: AlgorithmsEnum, dataset: DatasetsEnum, conf_path: str):
    print('Starting Train-Val-Test')
    print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')

    # ------ Run train and Val ------ #
    metrics_values, conf = run_train_val(alg, dataset, conf_path)
    # ------ Run test ------ #
    run_test(alg, dataset, conf)
