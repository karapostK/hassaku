import argparse
import json
import os
import socket
from pathlib import Path

from paramiko import SSHClient
from scp import SCPClient

import wandb
from algorithms.algorithms_utils import AlgorithmsEnum
from algorithms.naive_algs import PopularItems
from algorithms.sgd_alg import ExplainableCollaborativeFiltering
from data.data_utils import get_dataloader, DatasetsEnum
from data.dataset import TrainRecDataset, ECFTrainRecDataset
from eval.eval import evaluate_recommender_algorithm
from wandb_conf import ENTITY_NAME, PROJECT_NAME

parser = argparse.ArgumentParser(description='Start a test experiment')

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
print(best_run_model_path)

# Create base directory if absent
Path(os.path.dirname(best_run_model_path)).mkdir(parents=True, exist_ok=True)

current_host = socket.gethostname()

if current_host != best_run_host:
    print('Importing Model...')
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

            scp.get(remote_path=os.path.join(dir_path, best_run_model_path), local_path=best_run_model_path,
                    recursive=True)

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

# Model is now local
# Carry out Test
alg = AlgorithmsEnum[best_run_config['alg']]
dataset = DatasetsEnum[best_run_config['dataset']]
conf = best_run_config
print('Starting Test')
print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')

wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME, config=conf, tags=[alg.name, dataset.name],
           group=f'{alg.name} - {dataset.name} - test', name=conf['time_run'], job_type='test', reinit=True)

test_loader = get_dataloader(conf, 'test')

if alg.value == PopularItems:
    # Popular Items requires the popularity distribution over the items learned over the training data
    alg = alg.value.build_from_conf(conf, TrainRecDataset(conf['dataset_path']))
elif alg.value == ExplainableCollaborativeFiltering:
    alg = alg.value.build_from_conf(conf, ECFTrainRecDataset(conf['dataset_path']))
else:
    alg = alg.value.build_from_conf(conf, test_loader.dataset)

alg.load_model_from_path(conf['model_path'])

metrics_values = evaluate_recommender_algorithm(alg, test_loader, verbose=conf['running_settings']['batch_verbose'])

wandb.log(metrics_values, step=0)
wandb.finish()
