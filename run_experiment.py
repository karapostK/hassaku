import argparse
import logging

from algorithms.algorithms_utils import AlgorithmsEnum
from data.data_utils import DatasetsEnum
from experiment_helper import run_train_val, run_test, run_train_val_test

parser = argparse.ArgumentParser(description='Start an experiment')

parser.add_argument('--algorithm', '-a', type=str, help='Recommender Systems Algorithm',
                    choices=[alg.name for alg in AlgorithmsEnum])

parser.add_argument('--dataset', '-d', type=str, help='Recommender Systems Dataset',
                    choices=[dataset.name for dataset in DatasetsEnum], required=False, default='ml1m')
parser.add_argument('--conf_path', '-c', type=str, help='Path to the .yml containing the configuration')

parser.add_argument('--run_type', '-t', type=str, choices=['train_val', 'test', 'train_val_test'],
                    default='train_val_test',
                    help='Type of run to carry out among "Train + Val", "Test", and "Train + Val + Test"')
parser.add_argument('--log', type=str, default='WARNING')

parser.add_argument('--fair_method', '-f', type=str, choices=['none', 'weight_train', 'weight_train_val'],
                    default='none',
                    help='Which type of fairness strategy to adopt')

args = parser.parse_args()

alg = AlgorithmsEnum[args.algorithm]
dataset = DatasetsEnum[args.dataset]
conf_path = args.conf_path
run_type = args.run_type
log = args.log
fair_method = args.fair_method

if fair_method != 'none':
    print('Performing fair method')

logging.basicConfig(level=log)
if run_type == 'train_val':
    run_train_val(alg, dataset, conf_path, fair_method=fair_method)
elif run_type == 'test':
    run_test(alg, dataset, conf_path)
else:
    run_train_val_test(alg, dataset, conf_path, fair_method=fair_method)
