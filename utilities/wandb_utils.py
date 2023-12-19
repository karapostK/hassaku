import glob
import json
import os
import socket
from pathlib import Path

import wandb
from paramiko import SSHClient
from scp import SCPClient

from conf.conf_parser import parse_conf_file
from wandb_conf import ENTITY_NAME, PROJECT_NAME


def fetch_best_in_sweep(sweep_id, good_faith=True, preamble_path=None, project_base_directory: str = '..',
                        wandb_entitiy_name=ENTITY_NAME, wandb_project_name=PROJECT_NAME):
    """
    It returns the configuration of the best model of a specific sweep.
    However, since private wandb projects can only be accessed by 1 member, sharing of the models is basically impossible.
    The alternative, with good_faith=True it simply looks at the local directory with that specific sweep and hopes there is a model there.
    If there are multiple models then it raises an error.

    @param sweep_id:
    @param good_faith:  whether to look only at local folders or not
    @param preamble_path: if specified it will replace the part that precedes hassaku (e.g. /home/giovanni/hassaku... -> /something_else/hassaku/....
    @param project_base_directory: where is the project directory (either relative from where the code is running or in absolute path.
    @return:
    """
    if good_faith:
        sweep_path = glob.glob(f'{project_base_directory}/saved_models/*/sweeps/{sweep_id}')
        if len(sweep_path) > 1:
            raise ValueError('There should not be two sweeps with the same id')
        sweep_path = sweep_path[0]
        best_run_path = os.listdir(sweep_path)
        if len(best_run_path) > 1:
            raise ValueError('There are more than 1 runs in the project, which one is the best?')

        best_run_path = best_run_path[0]
        best_run_config = parse_conf_file(os.path.join(sweep_path, best_run_path, 'conf.yml'))

    else:
        api = wandb.Api()
        sweep = api.sweep(f"{wandb_entitiy_name}/{wandb_project_name}/{sweep_id}")

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
        local_path = os.path.join(project_base_directory, best_run_model_path)
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

                        scp.get(remote_path=os.path.join(dir_path, best_run_model_path),
                                local_path=os.path.dirname(local_path),
                                recursive=True)
            else:
                raise FileNotFoundError(f"The model should be local but it was not found! Path is: {local_path}")

    if preamble_path:
        pre, post = best_run_config['dataset_path'].split('hassaku/', 1)
        best_run_config['dataset_path'] = os.path.join(preamble_path, 'hassaku', post)
        pre, post = best_run_config['data_path'].split('hassaku/', 1)
        best_run_config['data_path'] = os.path.join(preamble_path, 'hassaku', post)

    # Running from non-main folder
    best_run_config['model_save_path'] = os.path.join(project_base_directory, best_run_config['model_save_path'])
    best_run_config['model_path'] = os.path.join(project_base_directory, best_run_config['model_path'])
    return best_run_config
