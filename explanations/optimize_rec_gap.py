import argparse
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import Subset

from algorithms.algorithms_utils import AlgorithmsEnum
from algorithms.naive_algs import PopularItems
from algorithms.sgd_alg import ECF
from conf.conf_parser import parse_conf_file
from data.data_utils import DatasetsEnum, get_dataloader
from data.dataloader import NegativeSampler, TrainDataLoader
from data.dataset import TrainRecDataset, ECFTrainRecDataset
from data.dataset import TrainUserRecDataset
from eval.eval import evaluate_recommender_algorithm
from explanations.fairness_utily import ConcatDataLoaders, RecGapLoss,fetch_best_in_sweep
from explanations.proto_algs_knobs import TunePrototypeRecModel

from train.trainer import Trainer
from utilities.utils import reproducible, generate_id
from wandb_conf import ENTITY_NAME

FAIR_SEED = 64
PROJECT_NAME = 'protochange'

parser = argparse.ArgumentParser(description='Start experiment')

parser.add_argument('--sweep_id', '-s', type=str, help='ID of the sweep')
parser.add_argument('--fairness_conf_path', '-c', type=str, help='Path to the fairness configuration file.')

args = parser.parse_args()

sweep_id = args.sweep_id
fairness_conf_path = args.fairness_conf_path

run_conf = fetch_best_in_sweep(sweep_id, good_faith=False, preamble_path='~/PycharmProjects')

# --- Fairness Configuration --- #
fair_conf = parse_conf_file(fairness_conf_path)
fair_conf['time_run'] = generate_id()
fair_conf['model_path'] = os.path.join(run_conf['model_path'], 'prototype_perturbations', fair_conf['time_run'])
fair_conf['running_settings']['seed'] = FAIR_SEED
fair_conf['dataset_path'] = run_conf['dataset_path']
fair_conf['start_group_1_in_batch'] = int(fair_conf['train_batch_size'] / 2)
Path(fair_conf['model_path']).mkdir(parents=True, exist_ok=True)

# --- Preparing the Model & Data --- #
alg = AlgorithmsEnum[run_conf['alg']]
dataset = DatasetsEnum[run_conf['dataset']]

reproducible(fair_conf['running_settings']['seed'])

train_dataset = TrainUserRecDataset(data_path=run_conf['dataset_path'], n_pos=fair_conf['pos_train'])
group_0_train_dataset = Subset(train_dataset, np.where(train_dataset.user_to_user_group == 0)[0])
group_1_train_dataset = Subset(train_dataset, np.where(train_dataset.user_to_user_group == 1)[0])

sampler = NegativeSampler(
    train_dataset=train_dataset,
    n_neg=fair_conf['neg_train'],
)

group_0_train_dataloader = TrainDataLoader(
    sampler,
    group_0_train_dataset,
    batch_size=int(fair_conf['train_batch_size'] / 2),
    shuffle=True,
    num_workers=int(fair_conf['running_settings']['train_n_workers'] / 2)
)

group_1_train_dataloader = TrainDataLoader(
    sampler,
    group_1_train_dataset,
    batch_size=int(fair_conf['train_batch_size'] / 2),
    shuffle=True,
    num_workers=int(fair_conf['running_settings']['train_n_workers'] / 2)
)

train_loader = ConcatDataLoaders(group_0_train_dataloader, group_1_train_dataloader)
val_loader = get_dataloader(fair_conf, 'val')

if alg.value == PopularItems:
    alg = alg.value.build_from_conf(run_conf, TrainRecDataset(run_conf['dataset_path']))
elif alg.value == ECF:
    alg = alg.value.build_from_conf(run_conf, ECFTrainRecDataset(run_conf['dataset_path']))
else:
    alg = alg.value.build_from_conf(run_conf, val_loader.dataset)

alg.load_model_from_path(run_conf['model_path'])
alg.requires_grad_(False)

# --- First Evaluation --- #

wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME, config=fair_conf, tags=[run_conf['alg'], run_conf['dataset']],
           group=f"{run_conf['alg']} - {run_conf['dataset']} - train/val", name=fair_conf['time_run'],
           job_type='train/val')

metrics_values = evaluate_recommender_algorithm(alg, val_loader, verbose=True)

wandb.log({**metrics_values,
           'weights': wandb.Histogram(torch.zeros(20))})  # TODO: logging the weights doesnt work for every step.

# --- Starting the experiment --- #

tuner = TunePrototypeRecModel(prototype_model=alg, entity_name=fair_conf['entity_name'], type_perturb='add')
tuner = tuner.to('cuda')

rec_loss = RecGapLoss.build_from_conf(fair_conf, None)
trainer = Trainer(tuner, train_loader, val_loader, rec_loss, fair_conf)

metrics_values = trainer.fit()
