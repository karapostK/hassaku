import argparse

from algorithms.algorithms_utils import AlgorithmsEnum
from data.data_utils import DatasetsEnum
from hyper_search.experiment_helper import start_hyper

parser = argparse.ArgumentParser(description='Start a Hyperparameter-optimization experiment')

parser.add_argument('--algorithm', '-a', type=str, help='Recommender Systems Algorithm',
                    choices=[alg.name for alg in AlgorithmsEnum])

parser.add_argument('--dataset', '-d', type=str, help='Recommender Systems Dataset',
                    choices=[dataset.name for dataset in DatasetsEnum], required=False, default='ml1m')
parser.add_argument('--conf_path', '-c', type=str,
                    help='Path to the .yml containing the configuration of the experiment.'
                         'NB. These values will be overridden by the hyper-parameter configuration.')
parser.add_argument('--n_samples', '-ns', type=int, default=100, required=False,
                    help='Number of hyperparameters configurations to sample')
parser.add_argument('--n_gpus', '-ng', type=float, default=0.2, required=False,
                    help='Number of gpus per trial (<= 1 values are possible)')
parser.add_argument('--n_concurrent', '-nc', type=int, default=None, required=False,
                    help='Number of allowed concurrent trials.')
parser.add_argument('--n_cpus', '-ncp', type=float, default=1, required=False,
                    help="Number of cpus per trails (<= 1 values are possible)")

args = parser.parse_args()

alg = AlgorithmsEnum[args.algorithm]
dataset = DatasetsEnum[args.dataset]
conf_path = args.conf_path
n_samples = args.n_samples
n_gpus = args.n_gpus
n_concurrent = args.n_concurrent
n_cpus = args.n_cpus

start_hyper(alg, dataset, conf_path, n_gpus=n_gpus, n_concurrent=n_concurrent, n_samples=n_samples,n_cpus=n_cpus)
