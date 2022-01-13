import argparse
import os

from experiment_helper import start_hyper#, start_multiple_hyper
from utilities.consts import SINGLE_SEED
from utilities.enums import RecAlgorithmsEnum

os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='Start an experiment')

parser.add_argument('--algorithm', '-a', type=str, help='Recommender System Algorithm',
                    choices=[alg.name for alg in RecAlgorithmsEnum])

parser.add_argument('--dataset', '-d', type=str, help='Recommender System Dataset',
                    choices=['amazon2014', 'ml-1m', 'lfm2b-1m'])

parser.add_argument('--multiple', '-mp', action='store_true', default=False, required=False)
parser.add_argument('--seed', '-s', type=int, default=SINGLE_SEED, required=False)

args = parser.parse_args()

alg = RecAlgorithmsEnum[args.algorithm]
dataset = args.dataset
multiple = args.multiple
seed = args.seed

#if multiple:
 #   start_multiple_hyper(alg, dataset)
#else:
start_hyper(alg, dataset, seed)
