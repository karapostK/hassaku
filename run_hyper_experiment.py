import argparse

from algorithms.algorithms_utils import AlgorithmsEnum
from data.data_utils import DatasetsEnum
from hyper_search.experiment_helper import start_hyper

parser = argparse.ArgumentParser(description='Start a Hyperparameter-optimization experiment')

parser.add_argument('--algorithm', '-a', type=str, help='Recommender Systems Algorithm',
                    choices=[alg.name for alg in AlgorithmsEnum])

parser.add_argument('--dataset', '-d', type=str, help='Recommender Systems Dataset',
                    choices=[dataset.name for dataset in DatasetsEnum], required=False, default='ml1m')
parser.add_argument('--data_path', '-dp', type=str,
                    help='Path to directory that contains the data.', required=True)
parser.add_argument('--n_samples', '-ns', type=int, default=50, required=False,
                    help='Number of hyperparameters configurations to sample')
parser.add_argument('--n_gpus', '-ngpu', type=float, default=0.2, required=False,
                    help='Number of gpus per trial (<= 1 values are possible)')
parser.add_argument('--n_cpus', '-ncpu', type=float, default=1, required=False,
                    help="Number of cpus per trails (<= 1 values are possible)")
parser.add_argument('--n_concurrent', '-nc', type=int, default=None, required=False,
                    help='Number of allowed concurrent trials.')

args = parser.parse_args()

alg = AlgorithmsEnum[args.algorithm]
dataset = DatasetsEnum[args.dataset]
data_path = args.data_path
n_samples = args.n_samples
n_gpus = args.n_gpus
n_concurrent = args.n_concurrent
n_cpus = args.n_cpus

start_hyper(alg, dataset, data_path, n_gpus=n_gpus, n_concurrent=n_concurrent, n_samples=n_samples, n_cpus=n_cpus)
