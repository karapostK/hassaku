import argparse
import os

import wandb
from algorithms.algorithms_utils import AlgorithmsEnum
from algorithms.naive_algs import PopularItems
from algorithms.sgd_alg import ECF, DeepMatrixFactorization
from data.data_utils import get_dataloader, DatasetsEnum
from data.dataset import TrainRecDataset, ECFTrainRecDataset
from eval.eval import evaluate_recommender_algorithm, FullEvaluator
from explanations.fairness_utily import fetch_best_in_sweep, build_user_and_item_tag_matrix, \
    build_user_and_item_pop_matrix, FullEvaluatorCalibrationDecorator

parser = argparse.ArgumentParser(description='Start a test experiment')

parser.add_argument('--sweep_id', '-s', type=str, help='ID of the sweep')
parser.add_argument('--measure_calibration', '-c', help='Whether to compute calibration metrics as well',
                    action='store_true', default=False)

args = parser.parse_args()

sweep_id = args.sweep_id
measure_calibration = args.measure_calibration

best_run_config = fetch_best_in_sweep(sweep_id, good_faith=False, project_base_directory='.',
                                      preamble_path='~/PycharmProjects', wandb_entitiy_name='karapost')

# Model is now local
# Carry out Test
alg = AlgorithmsEnum[best_run_config['alg']]
dataset = DatasetsEnum[best_run_config['dataset']]
conf = best_run_config
print('Starting Test')
print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')

wandb.init(project='protofair', entity='karapost', config=conf, tags=[alg.name, dataset.name],
           group=f'{alg.name} - {dataset.name} - test', name=conf['time_run'], job_type='test', reinit=True)

conf['running_settings']['eval_n_workers'] = 0
test_loader = get_dataloader(conf, 'test')

if alg.value == PopularItems or alg.value == DeepMatrixFactorization:
    # Popular Items requires the popularity distribution over the items learned over the training data
    # DeepMatrixFactorization also requires access to the training data
    alg = alg.value.build_from_conf(conf, TrainRecDataset(conf['dataset_path']))
elif alg.value == ECF:
    alg = alg.value.build_from_conf(conf, ECFTrainRecDataset(conf['dataset_path']))
else:
    alg = alg.value.build_from_conf(conf, test_loader.dataset)

alg.load_model_from_path(conf['model_path'])

evaluator = FullEvaluator(aggr_by_group=True, n_groups=test_loader.dataset.n_user_groups,
                          user_to_user_group=test_loader.dataset.user_to_user_group)

if measure_calibration:
    user_tag_mtx, item_tag_mtx = build_user_and_item_tag_matrix(os.path.join(conf['data_path'], conf['dataset']))
    user_pop_mtx, item_pop_mtx = build_user_and_item_pop_matrix(os.path.join(conf['data_path'], conf['dataset']))

    evaluator = FullEvaluatorCalibrationDecorator(evaluator, item_tag_mtx, user_tag_mtx, metric_name_prefix='tag')
    evaluator = FullEvaluatorCalibrationDecorator(evaluator, item_pop_mtx, user_pop_mtx, metric_name_prefix='pop')

metrics_values = evaluate_recommender_algorithm(alg, test_loader, evaluator,
                                                verbose=conf['running_settings']['batch_verbose'])

wandb.log(metrics_values, step=0)
wandb.finish()
