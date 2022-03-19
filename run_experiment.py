import argparse
import os

from consts.consts import SINGLE_SEED, NUM_SAMPLES
from consts.enums import RecAlgorithmsEnum, RecDatasetsEnum
from hyper_search.experiment_helper import start_multi_dataset, start_multiple_hyper, start_hyper

os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='Start an experiment')

parser.add_argument('--algorithm', '-a', type=str, help='Recommender System Algorithm',
                    choices=[alg.name for alg in RecAlgorithmsEnum])

parser.add_argument('--dataset', '-d', type=str, help='Recommender System Dataset',
                    choices=[dataset.name for dataset in RecDatasetsEnum], required=False, default='ml1m')

parser.add_argument('--seed', '-s', type=int, default=SINGLE_SEED, required=False,
                    help='Seed for initialization and sampling.')
parser.add_argument('--multiple', '-mp', action='store_true', default=False, required=False,
                    help='Whether to run all the experiments over a single dataset over the specified seeds (see consts.py). Overrides seeds.')
parser.add_argument('--every', '-e', action='store_true', default=False, required=False,
                    help='Whether to run all the experiments over all datasets and all seeds. Overrides multiple.')

parser.add_argument('--n_samples', '-ns', type=int, default=NUM_SAMPLES, required=False,
                    help='Number of hyperparameters configurations to sample')
parser.add_argument('--n_gpus', '-ng', type=float, default=0.2, required=False,
                    help='Number of gpus per trial (<= 1 values are possible)')
parser.add_argument('--n_concurrent', '-nc', type=int, default=None, required=False,
                    help='Number of allowed concurrent trials.')
parser.add_argument('--n_epochs', '-ne', type=int, default=100, required=False, help='Number of maximum epochs')
parser.add_argument('--n_workers', '-nw', type=int, default=2, required=False,
                    help='Number of workers for the training dataloader')

args = parser.parse_args()

alg = RecAlgorithmsEnum[args.algorithm]
dataset = RecDatasetsEnum[args.dataset]
seed = args.seed
multiple = args.multiple
every = args.every
n_samples = args.n_samples
n_gpus = args.n_gpus
n_concurrent = args.n_concurrent
n_epochs = args.n_epochs
n_workers = args.n_workers

if every:
    start_multi_dataset(alg, n_gpus=n_gpus, n_concurrent=n_concurrent, n_samples=n_samples, n_epochs=n_epochs,
                        n_workers=n_workers)
elif multiple:
    start_multiple_hyper(alg, dataset, n_gpus=n_gpus, n_concurrent=n_concurrent, n_samples=n_samples, n_epochs=n_epochs,
                         n_workers=n_workers)
else:
    start_hyper(alg, dataset, seed, n_gpus=n_gpus, n_concurrent=n_concurrent, n_samples=n_samples, n_epochs=n_epochs,
                n_workers=n_workers)
