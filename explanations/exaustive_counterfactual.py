import argparse
import os
import pickle

import torch
from tqdm import tqdm, trange

from algorithms.algorithms_utils import AlgorithmsEnum
from algorithms.sgd_alg import ECF
from data.data_utils import DatasetsEnum, get_dataloader
from data.dataset import ECFTrainRecDataset
from eval.eval import evaluate_recommender_algorithm, FullEvaluator
from explanations.fairness_utily import ALPHAS, fetch_best_in_sweep, build_user_and_item_tag_matrix, \
    build_user_and_item_pop_matrix, FullEvaluatorCalibrationDecorator
from explanations.proto_algs_knobs import PrototypeMultiply, PrototypeTuner

parser = argparse.ArgumentParser(description='Start an exhaustive counterfactual experiment')

parser.add_argument('--sweep_id', '-s', type=str, help='ID of the sweep', )
parser.add_argument('--batch_size', '-bs', type=int, help='Size of the evaluation batch', default=128)
parser.add_argument('--n_workers', '-nw', type=int, help='Number of workers for evaluation', default=8)

args = parser.parse_args()

sweep_id = args.sweep_id
batch_size = args.batch_size
n_workers = args.n_workers

conf = fetch_best_in_sweep(sweep_id, good_faith=True, preamble_path='~/PycharmProjects',
                           project_base_directory='..')

# --- Preparing the Model & Data --- #

alg = AlgorithmsEnum[conf['alg']]
dataset = DatasetsEnum[conf['dataset']]

conf['running_settings']['eval_n_workers'] = n_workers
conf['eval_batch_size'] = batch_size

val_loader = get_dataloader(conf, 'val')
test_loader = get_dataloader(conf, 'test')
assert len(val_loader) == len(test_loader), "The two dataloaders do not have the same lengths. Something went wrong."

n_users_groups = val_loader.dataset.n_user_groups
user_to_user_group = val_loader.dataset.user_to_user_group

user_tag_mtx, item_tag_mtx = build_user_and_item_tag_matrix(os.path.join(conf['data_path'], conf['dataset']))
user_pop_mtx, item_pop_mtx = build_user_and_item_pop_matrix(os.path.join(conf['data_path'], conf['dataset']))

val_evaluator = FullEvaluator(aggr_by_group=True, n_groups=n_users_groups, user_to_user_group=user_to_user_group)
test_evaluator = FullEvaluator(aggr_by_group=True, n_groups=n_users_groups, user_to_user_group=user_to_user_group)

# Adding Decorators (btw we don't have a copy function)
val_evaluator = FullEvaluatorCalibrationDecorator(val_evaluator, item_tag_mtx, user_tag_mtx, metric_name_prefix='tag')
val_evaluator = FullEvaluatorCalibrationDecorator(val_evaluator, item_pop_mtx, user_pop_mtx, metric_name_prefix='pop')

test_evaluator = FullEvaluatorCalibrationDecorator(test_evaluator, item_tag_mtx, user_tag_mtx, metric_name_prefix='tag')
test_evaluator = FullEvaluatorCalibrationDecorator(test_evaluator, item_pop_mtx, user_pop_mtx, metric_name_prefix='pop')

if alg.value == ECF:
    alg = alg.value.build_from_conf(conf, ECFTrainRecDataset(conf['dataset_path']))
else:
    alg = alg.value.build_from_conf(conf, val_loader.dataset)

alg.load_model_from_path(conf['model_path'])
alg = alg.to('cuda')

# --- Counting the Prototypes --- #

n_user_prototypes = 0
n_item_prototypes = 0
if conf['alg'] == "uprotomf":
    n_user_prototypes = alg.n_prototypes
elif conf['alg'] == "iprotomf":
    n_item_prototypes = alg.n_prototypes
elif conf['alg'] == 'acf':
    n_item_prototypes = n_user_prototypes = alg.n_anchors
elif conf['alg'] == 'ecf':
    n_user_prototypes = alg.n_clusters
elif conf['alg'] == 'uiprotomf':
    n_user_prototypes = alg.uprotomf.n_prototypes
    n_item_prototypes = alg.iprotomf.n_prototypes
else:
    ValueError('Model not recognized')

# --- Starting the exhaustive search --- #
# Dictionaries.
# Key is (entity_name,group,prototype_index,alpha)
# Values are the results from the FullEvaluatorCalibrationDecorator.

val_exhaustive_search_results_raw = dict()

test_exhaustive_search_results_raw = dict()

with torch.no_grad():
    # --- Loop over the entities --- #
    for n_prototypes, entity_name in tqdm(zip([n_user_prototypes, n_item_prototypes], ['user', 'item']), desc='Entity'):
        if n_prototypes == 0:
            continue
        # --- Loop over the prototypes --- #
        for prototype_index in trange(n_prototypes, desc='Prototype Index'):
            # --- Loop over the alpha values --- #
            for alpha in tqdm(ALPHAS, desc='Alphas'):
                # --- Loop over the user groups --- #
                for curr_group in trange(n_users_groups + 1, desc='Group'):
                    # NB. The last group is the group of all users.

                    # Setting the multiplication mask
                    lambda_mask = torch.ones((n_users_groups, n_prototypes), dtype=torch.float).to('cuda')
                    if curr_group == n_users_groups:
                        lambda_mask[:, prototype_index] = alpha
                    else:
                        lambda_mask[curr_group, prototype_index] = alpha

                    prototype_perturb = PrototypeMultiply(n_prototypes, n_users_groups, optimize_parameters=False,
                                                          default_parameters=lambda_mask, use_sigmoid=False)

                    prototype_tuner = PrototypeTuner(alg, entity_name, prototype_perturb,
                                                     user_to_user_group.to('cuda').long())

                    val_metrics = evaluate_recommender_algorithm(prototype_tuner, val_loader, val_evaluator,
                                                                 device='cuda', verbose=False)
                    test_metrics = evaluate_recommender_algorithm(prototype_tuner, test_loader, test_evaluator,
                                                                  device='cuda', verbose=False)

                    dict_key = (entity_name, -1 if curr_group == n_users_groups else curr_group, prototype_index, alpha)

                    val_exhaustive_search_results_raw[dict_key] = val_metrics

                    test_exhaustive_search_results_raw[dict_key] = test_metrics

        # --- Base Case --- #
        val_metrics = evaluate_recommender_algorithm(alg, val_loader, val_evaluator, device='cuda')
        test_metrics = evaluate_recommender_algorithm(alg, test_loader, test_evaluator, device='cuda')

        dict_key = (entity_name, -1, -1, 1)
        val_exhaustive_search_results_raw[dict_key] = val_metrics
        test_exhaustive_search_results_raw[dict_key] = test_metrics

# --- Save Dictionaries --- #

with open(f"./{conf['alg']}_{conf['dataset']}_val_results_raw.pkl", 'wb') as out_file:
    pickle.dump(val_exhaustive_search_results_raw, out_file)

with open(f"./{conf['alg']}_{conf['dataset']}_test_results_raw.pkl", 'wb') as out_file:
    pickle.dump(test_exhaustive_search_results_raw, out_file)
