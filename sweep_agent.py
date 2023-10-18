import glob
import os

import wandb
from algorithms.algorithms_utils import AlgorithmsEnum
from algorithms.base_classes import SGDBasedRecommenderAlgorithm, SparseMatrixBasedRecommenderAlgorithm
from conf.conf_parser import parse_conf, save_yaml
from data.data_utils import DatasetsEnum, get_dataloader
from data.dataset import TrainRecDataset
from eval.eval import evaluate_recommender_algorithm, FullEvaluator
from train.rec_losses import RecommenderSystemLoss
from train.trainer import Trainer
from utilities.utils import reproducible
from wandb_conf import KEEP_TOP_RUNS


def train_val_agent():
    # Initialization and gathering hyperparameters
    run = wandb.init(job_type='train/val')

    run_id = run.id
    project = run.project
    entity = run.entity
    sweep_id = run.sweep_id
    conf = {k: v for k, v in wandb.config.items() if k[0] != '_'}

    alg = AlgorithmsEnum[conf['alg']]
    dataset = DatasetsEnum[conf['dataset']]

    conf['sweep_id'] = sweep_id
    conf = parse_conf(conf, alg, dataset)

    # Updating wandb data
    run.tags += (alg.name, dataset.name)
    wandb.config.update(conf)

    print('Starting Train-Val')
    print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')
    print(f'Sweep ID is {sweep_id}')

    reproducible(conf['running_settings']['seed'])

    if issubclass(alg.value, SGDBasedRecommenderAlgorithm):

        train_loader = get_dataloader(conf, 'train')
        val_loader = get_dataloader(conf, 'val')

        alg = alg.value.build_from_conf(conf, train_loader.dataset)

        # Validation happens within the Trainer
        rec_loss = RecommenderSystemLoss.build_from_conf(conf, train_loader.dataset)
        trainer = Trainer(alg, train_loader, val_loader, rec_loss, conf)
        trainer.fit()

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
        wandb.log(metrics_values)
    elif alg in [AlgorithmsEnum.rand, AlgorithmsEnum.pop]:

        train_dataset = TrainRecDataset(conf['dataset_path'])
        val_loader = get_dataloader(conf, 'val')

        alg = alg.value.build_from_conf(conf, train_dataset)
        evaluator = FullEvaluator(aggr_by_group=True, n_groups=val_loader.dataset.n_user_groups,
                                  user_to_user_group=val_loader.dataset.user_to_user_group)
        metrics_values = evaluate_recommender_algorithm(alg, val_loader, evaluator,
                                                        verbose=conf['running_settings']['batch_verbose'])
        wandb.log(metrics_values)
    else:
        raise ValueError(f'Training for {alg.value} has been not implemented')
    save_yaml(conf['model_path'], conf)

    # To reduce space consumption. Check if the run is in the top-10 best. If not, delete the model.

    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    top_runs = api.runs(path=f'{entity}/{project}',
                        per_page=KEEP_TOP_RUNS,
                        order=sweep.order,
                        filters={"$and": [{"sweep": f"{sweep_id}"}]}
                        )[:KEEP_TOP_RUNS]
    top_runs_ids = {x.id for x in top_runs}

    if run_id not in top_runs_ids:
        print(f'Run {run_id} is not in the top-{KEEP_TOP_RUNS}.')
        print(f'Model will be deleted')
        # Delete Model

        alg_model_path = os.path.join(conf['model_path'], 'model.*')
        alg_model_files = glob.glob(alg_model_path)
        for alg_model_file in alg_model_files:
            os.remove(alg_model_file)

    wandb.finish()


train_val_agent()
