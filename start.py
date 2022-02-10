import argparse
import os

from experiment_helper import start_hyper, start_multiple_hyper, start_multi_dataset
from utilities.consts import SINGLE_SEED, NUM_SAMPLES
from utilities.enums import RecAlgorithmsEnum, RecDatasetsEnum

os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='Start an experiment')

parser.add_argument('--algorithm', '-a', type=str, help='Recommender System Algorithm',
                    choices=[alg.name for alg in RecAlgorithmsEnum])

parser.add_argument('--dataset', '-d', type=str, help='Recommender System Dataset',
                    choices=[dataset.name for dataset in RecDatasetsEnum])

parser.add_argument('--seed', '-s', type=int, default=SINGLE_SEED, required=False,
                    help='Seed for initialization and sampling')
parser.add_argument('--multiple', '-mp', action='store_true', default=False, required=False,
                    help='Whether to run all the experiments over a single dataset over the specified seeds (see consts.py). Overrides seeds.')
parser.add_argument('--every', '-e', action='store_true', default=False, required=False,
                    help='Whether to run all the experiments over all datasets and all seeds. Overrides multiple.')

parser.add_argument('--n_samples', '-ns', type=int, default=NUM_SAMPLES, required=False,
                    help='Number of hyperparameters configurations to sample')
parser.add_argument('--n_gpus', '-g', type=float, default=0.2, required=False,
                    help='Number of gpus per trial (<= 1 values are possible)')
parser.add_argument('--n_concurrent', '-c', type=int, default=None, required=False,
                    help='Number of allowed concurrent trials.')

args = parser.parse_args()

alg = RecAlgorithmsEnum[args.algorithm]
dataset = RecDatasetsEnum[args.dataset]
seed = args.seed
multiple = args.multiple
every = args.every
n_samples = args.n_samples
n_gpus = args.n_gpus
n_concurrent = args.n_concurrent

if every:
    start_multi_dataset(alg, n_gpus=n_gpus, n_concurrent=n_concurrent, n_samples=n_samples)
elif multiple:
    start_multiple_hyper(alg, dataset, n_gpus=n_gpus, n_concurrent=n_concurrent, n_samples=n_samples)
else:
    start_hyper(alg, dataset, seed, n_gpus=n_gpus, n_concurrent=n_concurrent, n_samples=n_samples)
