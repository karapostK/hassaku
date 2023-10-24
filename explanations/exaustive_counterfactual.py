import argparse
import os
import pickle

import pandas as pd
import torch
from tqdm import tqdm, trange

from algorithms.algorithms_utils import AlgorithmsEnum
from algorithms.sgd_alg import ECF
from data.data_utils import DatasetsEnum, get_dataloader
from data.dataset import ECFTrainRecDataset
from explanations.fairness_utily import ALPHAS, fetch_best_in_sweep, FullEvaluatorCalibration, \
    build_user_and_item_tag_matrix, multiply_mask, compute_rec_gap

parser = argparse.ArgumentParser(description='Start an exhaustive counterfactual experiment')

parser.add_argument('--sweep_id', '-s', type=str, help='ID of the sweep', )
parser.add_argument('--batch_size', '-bs', type=int, help='Size of the evaluation batch', default=128)

args = parser.parse_args()

sweep_id = args.sweep_id
batch_size = args.batch_size

conf = fetch_best_in_sweep(sweep_id, good_faith=True, preamble_path='~/PycharmProjects',
                           project_base_directory='..')

# --- Preparing the Model & Data --- #

alg = AlgorithmsEnum[conf['alg']]
dataset = DatasetsEnum[conf['dataset']]

conf['running_settings']['eval_n_workers'] = 0
conf['eval_batch_size'] = batch_size

val_loader = get_dataloader(conf, 'val')
test_loader = get_dataloader(conf, 'test')
assert len(val_loader) == len(test_loader), "The two dataloaders do not have the same lengths. Something went wrong."

n_users_groups = val_loader.dataset.n_user_groups
user_to_user_group = val_loader.dataset.user_to_user_group

user_tag_mtx, item_tag_mtx = build_user_and_item_tag_matrix(os.path.join(conf['data_path'], conf['dataset']))

val_evaluator = FullEvaluatorCalibration(aggr_by_group=True, n_groups=n_users_groups,
                                         user_to_user_group=user_to_user_group, item_tag_matrix=item_tag_mtx,
                                         train_user_tag_mtx=user_tag_mtx)
test_evaluator = FullEvaluatorCalibration(aggr_by_group=True, n_groups=n_users_groups,
                                          user_to_user_group=user_to_user_group, item_tag_matrix=item_tag_mtx,
                                          train_user_tag_mtx=user_tag_mtx)

if alg.value == ECF:
    alg = alg.value.build_from_conf(conf, ECFTrainRecDataset(conf['dataset_path']))
else:
    alg = alg.value.build_from_conf(conf, val_loader.dataset)

alg.load_model_from_path(conf['model_path'])
alg = alg.to('cuda')

# --- Counting the Prototypes --- #

n_prototypes = None
if conf['alg'] == "uprotomf" or conf['alg'] == "iprotomf":
    n_prototypes = alg.n_prototypes
elif conf['alg'] == 'acf':
    n_prototypes = 2 * alg.n_anchors  # Dropping the weights for the users comes first
elif conf['alg'] == 'ecf':
    n_prototypes = 2 * alg.n_clusters  # Dropping the weights for the users comes first
elif conf['alg'] == 'uiprotomf':
    n_prototypes = alg.uprotomf.n_prototypes + alg.iprotomf.n_prototypes
else:
    ValueError('Model not recognized')

# --- Starting the exhaustive search --- #
# Dictionaries.
# Key is (prototype_index,alpha)
# Values are the results from the FullEvaluatorCalibration.

val_exhaustive_search_results = dict()
val_exhaustive_search_results_raw = dict()

test_exhaustive_search_results = dict()
test_exhaustive_search_results_raw = dict()

with torch.no_grad():
    for prototype_index in trange(n_prototypes, desc='Prototype Index'):
        for alpha in tqdm(ALPHAS, desc='Alphas'):

            # Setting the multiplication mask
            multiplication_mask = torch.ones(n_prototypes, dtype=torch.float16).to('cuda')
            multiplication_mask[prototype_index] = alpha

            i_idxs = torch.arange(val_loader.dataset.n_items).to('cuda')

            # Modifying the item representation if needed
            if conf['alg'] == "iprotomf":
                i_repr_middle = alg.get_item_representations_pre_tune(i_idxs)
                i_repr_middle = multiply_mask(i_repr_middle, multiplication_mask)
                i_repr_new = alg.get_item_representations_post_tune(i_repr_middle)
            elif conf['alg'] == 'uiprotomf':
                if prototype_index - alg.uprotomf.n_prototypes >= 0:
                    # The first entries in the multiplication mask are for UProtoMF, the latter for IProtoMF
                    sub_mask = multiplication_mask[alg.uprotomf.n_prototypes:]

                    i_repr_middle = alg.get_item_representations_pre_tune(i_idxs)
                    i_repr_middle = (multiply_mask(i_repr_middle[0], sub_mask), i_repr_middle[1])
                    i_repr_new = alg.get_item_representations_post_tune(i_repr_middle)
                else:
                    i_repr_new = alg.get_item_representations(i_idxs)
            elif conf['alg'] == 'acf':
                if prototype_index - alg.n_anchors >= 0:
                    sub_mask = multiplication_mask[alg.n_anchors:]

                    i_repr_middle = alg.get_item_representations_pre_tune(i_idxs)
                    i_repr_middle = multiply_mask(i_repr_middle, sub_mask)
                    i_repr_new = alg.get_item_representations_post_tune(i_repr_middle)
                else:
                    i_repr_new = alg.get_item_representations(i_idxs)
            elif conf['alg'] == 'ecf':
                if prototype_index - alg.n_clusters >= 0:
                    sub_mask = multiplication_mask[alg.n_clusters:]

                    i_repr_middle = alg.get_item_representations_pre_tune(None)
                    i_repr_middle = multiply_mask(i_repr_middle, sub_mask)
                    i_repr_new = alg.get_item_representations_post_tune(i_repr_middle)
                else:
                    i_repr_new = alg.get_item_representations(i_idxs)
            else:
                i_repr_new = alg.get_item_representations(i_idxs)

            for (u_idxs, _, val_labels), (_, _, test_labels) in zip(val_loader, test_loader):
                u_idxs = u_idxs.to('cuda')

                val_labels = val_labels.to('cuda')
                test_labels = test_labels.to('cuda')

                val_mask = torch.tensor(val_loader.dataset.exclude_data[u_idxs.cpu()].A).to('cuda')
                test_mask = torch.tensor(test_loader.dataset.exclude_data[u_idxs.cpu()].A).to('cuda')

                if conf['alg'] == 'uprotomf':
                    u_repr_middle = alg.get_user_representations_pre_tune(u_idxs)
                    u_repr_middle = multiply_mask(u_repr_middle, multiplication_mask)
                    u_repr_new = alg.get_user_representations_post_tune(u_repr_middle)
                elif conf['alg'] == 'uiprotomf':
                    if prototype_index - alg.uprotomf.n_prototypes < 0:
                        # The first entries in the multiplication mask are for UProtoMF, the latter for IProtoMF
                        sub_mask = multiplication_mask[:alg.uprotomf.n_prototypes]

                        u_repr_middle = alg.get_user_representations_pre_tune(u_idxs)
                        u_repr_middle = (multiply_mask(u_repr_middle[0], sub_mask), u_repr_middle[1])
                        u_repr_new = alg.get_user_representations_post_tune(u_repr_middle)
                    else:
                        u_repr_new = alg.get_user_representations(u_idxs)
                elif conf['alg'] == 'acf':
                    if prototype_index - alg.n_anchors < 0:
                        sub_mask = multiplication_mask[:alg.n_anchors]
                        u_repr_middle = alg.get_user_representations_pre_tune(u_idxs)
                        u_repr_middle = multiply_mask(u_repr_middle, sub_mask)
                        u_repr_new = alg.get_user_representations_post_tune(u_repr_middle)
                    else:
                        u_repr_new = alg.get_user_representations(u_idxs)
                elif conf['alg'] == 'ecf':
                    if prototype_index - alg.n_clusters < 0:
                        sub_mask = multiplication_mask[:alg.n_clusters]

                        u_repr_middle = alg.get_user_representations_pre_tune(u_idxs)
                        u_repr_middle = (multiply_mask(u_repr_middle[0], sub_mask), u_repr_middle[1])
                        u_repr_new = alg.get_user_representations_post_tune(u_repr_middle)
                    else:
                        u_repr_new = alg.get_user_representations(u_idxs)

                else:
                    u_repr_new = alg.get_user_representations(u_idxs)

                # Predictions
                out = alg.combine_user_item_representations(u_repr_new, i_repr_new)

                out_val = out
                out_test = out_val.clone()

                out_val[val_mask] = -torch.inf
                out_test[test_mask] = -torch.inf

                val_evaluator.eval_batch(u_idxs, out_val, val_labels)
                test_evaluator.eval_batch(u_idxs, out_test, test_labels)

            val_metrics = val_evaluator.get_results()
            test_metrics = test_evaluator.get_results()

            dict_key = (prototype_index, alpha)
            val_dict_value = list()
            test_dict_value = list()
            for dict_value, result_dict in zip([val_dict_value, test_dict_value], [val_metrics, test_metrics]):
                for metric_name in ['ndcg@10', 'precision@10', 'recall@10', 'hellinger_distance@100',
                                    'jensen_shannon_distance@100', 'kl_divergence@100']:
                    dict_value += [
                        result_dict[metric_name],
                        result_dict['group_0_' + metric_name],
                        result_dict['group_1_' + metric_name],
                        compute_rec_gap(result_dict, metric_name),
                    ]

            val_exhaustive_search_results[dict_key] = val_dict_value
            val_exhaustive_search_results_raw[dict_key] = val_metrics

            test_exhaustive_search_results[dict_key] = test_dict_value
            test_exhaustive_search_results_raw[dict_key] = test_metrics

    # --- Base Case --- #

    i_idxs = torch.arange(val_loader.dataset.n_items).to('cuda')
    i_repr_new = alg.get_item_representations(i_idxs)

    for (u_idxs, _, val_labels), (_, _, test_labels) in zip(val_loader, test_loader):
        u_idxs = u_idxs.to('cuda')

        val_labels = val_labels.to('cuda')
        test_labels = test_labels.to('cuda')

        val_mask = torch.tensor(val_loader.dataset.exclude_data[u_idxs.cpu()].A).to('cuda')
        test_mask = torch.tensor(test_loader.dataset.exclude_data[u_idxs.cpu()].A).to('cuda')

        u_repr_new = alg.get_user_representations(u_idxs)
        out = alg.combine_user_item_representations(u_repr_new, i_repr_new)

        out_val = out
        out_test = out_val.clone()

        out_val[val_mask] = -torch.inf
        out_test[test_mask] = -torch.inf

        val_evaluator.eval_batch(u_idxs, out_val, val_labels)
        test_evaluator.eval_batch(u_idxs, out_test, test_labels)

    val_metrics = val_evaluator.get_results()
    test_metrics = test_evaluator.get_results()

    dict_key = (-1, 1)
    val_dict_value = list()
    test_dict_value = list()
    for dict_value, result_dict in zip([val_dict_value, test_dict_value], [val_metrics, test_metrics]):
        column_names = []  # Saving column names, so it is easy to provide column names for the pandas dataframe
        for metric_name in ['ndcg@10', 'precision@10', 'recall@10', 'hellinger_distance@100',
                            'jensen_shannon_distance@100', 'kl_divergence@100']:
            dict_value += [
                result_dict[metric_name],
                result_dict['group_0_' + metric_name],
                result_dict['group_1_' + metric_name],
                compute_rec_gap(result_dict, metric_name),
            ]
            column_names += [metric_name, 'group_0_' + metric_name, 'group_1_' + metric_name, f'recgap({metric_name})']

    val_exhaustive_search_results[dict_key] = val_dict_value
    val_exhaustive_search_results_raw[dict_key] = val_metrics

    test_exhaustive_search_results[dict_key] = test_dict_value
    test_exhaustive_search_results_raw[dict_key] = test_metrics

# --- Save Dictionaries --- #

with open(f"./{conf['alg']}_{conf['dataset']}_val_results_raw.pkl", 'wb') as out_file:
    pickle.dump(val_exhaustive_search_results_raw, out_file)

with open(f"./{conf['alg']}_{conf['dataset']}_test_results_raw.pkl", 'wb') as out_file:
    pickle.dump(test_exhaustive_search_results_raw, out_file)

val_df = pd.DataFrame.from_dict(val_exhaustive_search_results, orient='index', columns=column_names)
val_df.to_csv(f"./{conf['alg']}_{conf['dataset']}_val_results.csv")

test_df = pd.DataFrame.from_dict(test_exhaustive_search_results, orient='index', columns=column_names)
test_df.to_csv(f"./{conf['alg']}_{conf['dataset']}_test_results.csv")
