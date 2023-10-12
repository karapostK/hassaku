import argparse
import json
import os
import pickle
import socket
from pathlib import Path

import pandas as pd
import scipy.sparse
import torch
import wandb
from paramiko import SSHClient
from scp import SCPClient
from tqdm import trange, tqdm

from algorithms.algorithms_utils import AlgorithmsEnum
from algorithms.naive_algs import PopularItems
from algorithms.sgd_alg import ECF
from data.data_utils import DatasetsEnum, get_dataloader
from data.dataset import TrainRecDataset, ECFTrainRecDataset
from eval.eval import FullEvaluator
from explanations.fairness_utily import ALPHAS, multiply_mask, TOP_K_RECOMMENDATIONS, compute_rec_gap, \
    hellinger_distance, jensen_shannon_distance, kl_divergence
from wandb_conf import ENTITY_NAME, PROJECT_NAME

parser = argparse.ArgumentParser(description='Start an exhaustive counterfactual experiment')

parser.add_argument('--sweep_id', '-s', type=str, help='ID of the sweep', )

args = parser.parse_args()

sweep_id = args.sweep_id

api = wandb.Api()
sweep = api.sweep(f"{ENTITY_NAME}/{PROJECT_NAME}/{sweep_id}")

best_run = sweep.best_run()
best_run_host = best_run.metadata['host']
best_run_config = json.loads(best_run.json_config)
if '_items' in best_run_config:
    best_run_config = best_run_config['_items']['value']
else:
    best_run_config = {k: v['value'] for k, v in best_run_config.items()}

best_run_model_path = best_run_config['model_path']
print('Best Run Model Path: ', best_run_model_path)

# Create base directory if absent
local_path = os.path.join('..', best_run_model_path)
current_host = socket.gethostname()

if not os.path.isdir(local_path):
    Path(local_path).mkdir(parents=True, exist_ok=True)

    if current_host != best_run_host:
        print(f'Importing Model from {best_run_host}')
        # Moving the best model to local directory
        # N.B. Assuming same username
        with SSHClient() as ssh:
            ssh.load_system_host_keys()
            ssh.connect(best_run_host)

            with SCPClient(ssh.get_transport()) as scp:
                # enoughcool4hardcoding
                dir_path = "hassaku"
                if best_run_host == 'passionpit.cp.jku.at':
                    dir_path = os.path.join(dir_path, "PycharmProjects")

                scp.get(remote_path=os.path.join(dir_path, best_run_model_path), local_path=os.path.dirname(local_path),
                        recursive=True)
    else:
        raise FileNotFoundError(f"The model should be local but it was not found! Path is: {local_path}")

# Adjusting dataset_path configuration for testing #enoughcool4hardcoding
if current_host == 'passionpit.cp.jku.at' and best_run_host != 'passionpit.cp.jku.at':
    # From RK to passionpit, adding PycharmProjects
    dataset_path = Path(best_run_config['dataset_path'])
    path_parts = list(dataset_path.parts[:3]) + ['PycharmProjects'] + list(dataset_path.parts[3:])
    best_run_config['dataset_path'] = os.path.join(*path_parts)
elif current_host != 'passionpit.cp.jku.at' and best_run_host == 'passionpit.cp.jku.at':
    # From passionpit to RK, removing PycharmProjects
    dataset_path = Path(best_run_config['dataset_path'])
    path_parts = list(dataset_path.parts[:3]) + list(dataset_path.parts[4:])
    best_run_config['dataset_path'] = os.path.join(*path_parts)
# --- Preparing the Model --- #

conf = best_run_config
alg = AlgorithmsEnum[conf['alg']]
dataset = DatasetsEnum[conf['dataset']]

conf['running_settings']['eval_n_workers'] = 4
conf['eval_batch_size'] = 64
val_loader = get_dataloader(conf, 'val')
val_evaluator = FullEvaluator(aggr_by_group=True, n_groups=val_loader.dataset.n_user_groups,
                              user_to_user_group=val_loader.dataset.user_to_user_group)

if alg.value == PopularItems:
    # Popular Items requires the popularity distribution over the items learned over the training data
    alg = alg.value.build_from_conf(conf, TrainRecDataset(conf['dataset_path']))
elif alg.value == ECF:
    alg = alg.value.build_from_conf(conf, ECFTrainRecDataset(conf['dataset_path']))
else:
    alg = alg.value.build_from_conf(conf, val_loader.dataset)

alg.load_model_from_path(local_path)
alg = alg.to('cuda')

# --- Preparing the Tag Matrix  --- #

assert conf['dataset'] in ['ml1m', 'lfm2b2020'], "Dataset is not valid!"
tag_csv = pd.read_csv(f"../data/{conf['dataset']}/processed_dataset/tag_idxs.csv")
item_tag_idxs_csv = pd.read_csv(f"../data/{conf['dataset']}/processed_dataset/item_tag_idxs.csv")

n_tags = len(tag_csv)
tag_matrix = torch.zeros(size=(alg.n_items, len(tag_csv)), dtype=torch.float16)
tag_matrix[[item_tag_idxs_csv.item_idx, item_tag_idxs_csv.tag_idx]] = 1.

# Normalizing row-wise
tag_matrix /= tag_matrix.sum(-1)[:, None]

# --- Preparing the Training History Distribution --- #

train_data = pd.read_csv(f"../data/{conf['dataset']}/processed_dataset/listening_history_train.csv")[
    ['user_idx', 'item_idx']]
train_mtx = scipy.sparse.csr_matrix(
    (torch.ones(len(train_data), dtype=torch.int16), (train_data.user_idx, train_data.item_idx)),
    shape=(alg.n_users, alg.n_items))

train_users_tag_frequencies = train_mtx @ tag_matrix
user_n_items = train_mtx.sum(-1).A
train_users_tag_frequencies /= user_n_items

tag_matrix = tag_matrix.to('cuda')
train_users_tag_frequencies = torch.tensor(train_users_tag_frequencies, device='cuda')

# Smoothening the train frequency distribution (eq. 7 in Calibrated Recommendation by H.Steck) beta = 0.01
train_users_tag_frequencies = .01 / n_tags + (1 - .01) * train_users_tag_frequencies

del train_data, train_mtx, user_n_items, tag_csv, item_tag_idxs_csv
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

# --- Preparing User to User Group assignments --- #

user_to_user_group = val_loader.dataset.user_to_user_group.to('cuda')
n_users_group_1 = user_to_user_group.sum()
n_users_group_0 = len(user_to_user_group) - n_users_group_1

# --- Starting the exhaustive search --- #
exhaustive_search_results = dict()
exhaustive_search_results_raw = dict()

with torch.no_grad():
    for prototype_index in trange(n_prototypes, desc='Prototype Index'):
        for alpha in tqdm(ALPHAS, desc='Alphas'):

            # Setting the multiplication mask
            multiplication_mask = torch.ones(n_prototypes, dtype=torch.float16).to('cuda')
            multiplication_mask[prototype_index] = alpha

            # Accumulating per-group frequencies
            all_hellinger = torch.zeros(1, dtype=torch.float, device='cuda')
            group_0_hellinger = torch.zeros(1, dtype=torch.float, device='cuda')
            group_1_hellinger = torch.zeros(1, dtype=torch.float, device='cuda')
            all_jsd = torch.zeros(1, dtype=torch.float, device='cuda')
            group_0_jsd = torch.zeros(1, dtype=torch.float, device='cuda')
            group_1_jsd = torch.zeros(1, dtype=torch.float, device='cuda')
            all_kl = torch.zeros(1, dtype=torch.float, device='cuda')
            group_0_kl = torch.zeros(1, dtype=torch.float, device='cuda')
            group_1_kl = torch.zeros(1, dtype=torch.float, device='cuda')

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

                    i_repr_middle = alg.get_item_representations_pre_tune()
                    i_repr_middle = multiply_mask(i_repr_middle, sub_mask)
                    i_repr_new = alg.get_item_representations_post_tune(i_repr_middle)
                else:
                    i_repr_new = alg.get_item_representations(i_idxs)
            else:
                i_repr_new = alg.get_item_representations(i_idxs)

            for u_idxs, _, val_labels in val_loader:
                u_idxs = u_idxs.to('cuda')
                val_labels = val_labels.to('cuda')
                val_mask = torch.tensor(val_loader.dataset.exclude_data[u_idxs.cpu()].A).to('cuda')

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

                out[val_mask] = -torch.inf

                val_evaluator.eval_batch(u_idxs, out, val_labels)

                batch_user_top_items = out.topk(TOP_K_RECOMMENDATIONS).indices
                batch_user_top_items_tags = tag_matrix[batch_user_top_items]  # [batch_size, top_k, n_tags]

                batch_user_tags_frequency = batch_user_top_items_tags.sum(1)  # [batch_size, n_tags]
                batch_user_tags_frequency /= TOP_K_RECOMMENDATIONS

                batch_user_train_tags_frequency = train_users_tag_frequencies[u_idxs]
                # Smoothening the batch frequency (Eq.5 in Calibrated Recommendation by H. Steck) alpha =.01
                batch_user_tags_frequency = .01 * batch_user_train_tags_frequency + (
                        1 - .01) * batch_user_tags_frequency

                # JSD, Hellinger & KL

                batch_hellinger = hellinger_distance(batch_user_tags_frequency,
                                                     batch_user_train_tags_frequency)  # [batch_size]
                batch_jsd = jensen_shannon_distance(batch_user_tags_frequency,
                                                    batch_user_train_tags_frequency)  # [batch_size]

                batch_kl = kl_divergence(batch_user_train_tags_frequency, batch_user_tags_frequency)

                all_jsd += batch_jsd.sum()
                all_hellinger += batch_hellinger.sum()
                all_kl += batch_kl.sum()

                batch_user_group = user_to_user_group[u_idxs]
                # Assuming two groups
                is_group_1_mask = batch_user_group.bool()

                group_0_hellinger += batch_hellinger[~is_group_1_mask].sum()
                group_1_hellinger += batch_hellinger[is_group_1_mask].sum()

                group_0_jsd += batch_jsd[~is_group_1_mask].sum()
                group_1_jsd += batch_jsd[is_group_1_mask].sum()

                group_0_kl += batch_kl[~is_group_1_mask].sum()
                group_1_kl += batch_kl[is_group_1_mask].sum()

            group_0_hellinger /= n_users_group_0
            group_0_jsd /= n_users_group_0
            group_0_kl /= n_users_group_0
            group_1_hellinger /= n_users_group_1
            group_1_jsd /= n_users_group_1
            group_1_kl /= n_users_group_1

            all_jsd /= (n_users_group_1 + n_users_group_0)
            all_hellinger /= (n_users_group_1 + n_users_group_0)
            all_kl /= (n_users_group_1 + n_users_group_0)

            val_result_dict = val_evaluator.get_results()
            val_result_dict['hellinger_distance'] = all_hellinger.item()
            val_result_dict['group_0_hellinger_distance'] = group_0_hellinger.item()
            val_result_dict['group_1_hellinger_distance'] = group_1_hellinger.item()
            val_result_dict['jensen_shannon_distance'] = all_jsd.item()
            val_result_dict['group_0_jensen_shannon_distance'] = group_0_jsd.item()
            val_result_dict['group_1_jensen_shannon_distance'] = group_1_jsd.item()
            val_result_dict['kl_divergence'] = all_kl.item()
            val_result_dict['group_0_kl_divergence'] = group_0_kl.item()
            val_result_dict['group_1_kl_divergence'] = group_1_kl.item()

            # Placing results into a dictionary

            dict_key = (prototype_index, alpha)
            dict_value = [val_result_dict['ndcg@10'],
                          val_result_dict['group_0_ndcg@10'],
                          val_result_dict['group_1_ndcg@10'],
                          compute_rec_gap(val_result_dict),
                          val_result_dict['hellinger_distance'],
                          val_result_dict['group_0_hellinger_distance'],
                          val_result_dict['group_1_hellinger_distance'],
                          compute_rec_gap(val_result_dict, 'hellinger_distance'),
                          val_result_dict['jensen_shannon_distance'],
                          val_result_dict['group_0_jensen_shannon_distance'],
                          val_result_dict['group_1_jensen_shannon_distance'],
                          compute_rec_gap(val_result_dict, 'jensen_shannon_distance'),
                          val_result_dict['kl_divergence'],
                          val_result_dict['group_0_kl_divergence'],
                          val_result_dict['group_1_kl_divergence'],
                          compute_rec_gap(val_result_dict, 'kl_divergence')
                          ]

            exhaustive_search_results[dict_key] = dict_value
            exhaustive_search_results_raw[dict_key] = [val_result_dict]

    # Last Update considers no changes

# Save dictionaries
with open(f"./{conf['alg']}_{conf['dataset']}_results_raw.pkl", 'wb') as out_file:
    pickle.dump(exhaustive_search_results_raw, out_file)

df = pd.DataFrame.from_dict(exhaustive_search_results, orient='index',
                            columns=['ndcg@10',
                                     'group_0_ndcg@10',
                                     'group_1_ndcg@10',
                                     'recgap(ndcg@10)',
                                     'hellinger_distance',
                                     'group_0_hellinger_distance',
                                     'group_1_hellinger_distance',
                                     'recgap(hellinger_distance)',
                                     'jensen_shannon_distance',
                                     'group_0_jensen_shannon_distance',
                                     'group_1_jensen_shannon_distance',
                                     'recgap(jensen_shannon_distance)',
                                     'kl_divergence',
                                     'group_0_kl_divergence',
                                     'group_1_kl_divergence',
                                     'recgap(kl_divergence)'])

df.to_csv(f"./{conf['alg']}_{conf['dataset']}_results.csv")
