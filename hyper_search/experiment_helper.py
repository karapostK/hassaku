import warnings

import wandb
from ray import air
from ray import tune
from ray.air import session
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.search.hyperopt import HyperOptSearch

from algorithms.algorithms_utils import AlgorithmsEnum
from algorithms.base_classes import SGDBasedRecommenderAlgorithm, SparseMatrixBasedRecommenderAlgorithm
from algorithms.naive_algs import PopularItems
from algorithms.sgd_alg import ECF, DeepMatrixFactorization
from conf.conf_parser import parse_conf
from conf.conf_parser import save_yaml
from data.data_utils import DatasetsEnum, get_dataloader
from data.dataset import TrainRecDataset, ECFTrainRecDataset
from eval.eval import evaluate_recommender_algorithm, FullEvaluator
from hyper_search.hyper_params import alg_data_param
from hyper_search.utils import KeepOnlyTopModels
from train.rec_losses import RecommenderSystemLossesEnum
from train.trainer import Trainer
from utilities.utils import generate_id, reproducible
from wandb_conf import PROJECT_NAME, ENTITY_NAME, WANDB_API_KEY_PATH


def tune_training(conf: dict):
    """
    Function executed by ray tune. It corresponds to a single trial.
    """
    conf['_in_tune'] = True
    reproducible(conf['running_settings']['seed'])
    conf['model_path'] = tune.get_trial_dir()

    alg = AlgorithmsEnum[conf['alg']]

    if issubclass(alg.value, SGDBasedRecommenderAlgorithm):

        train_loader = get_dataloader(conf, 'train')
        val_loader = get_dataloader(conf, 'val')

        rec_loss = RecommenderSystemLossesEnum[conf['rec_loss']]

        alg = alg.value.build_from_conf(conf, train_loader.dataset)
        rec_loss = rec_loss.value.build_from_conf(conf, train_loader.dataset)

        trainer = Trainer(alg, train_loader, val_loader, rec_loss, conf)

        # Validation happens within the Trainer
        metrics_values = trainer.fit()
        save_yaml(conf['model_path'], conf)

    elif issubclass(alg.value, SparseMatrixBasedRecommenderAlgorithm):
        train_dataset = TrainRecDataset(conf['dataset_path'])
        val_loader = get_dataloader(conf, 'val')

        alg = alg.value.build_from_conf(conf, train_dataset)

        # -- Training --
        alg.fit(train_dataset.sampling_matrix)
        # -- Validation --
        evaluator = FullEvaluator(aggr_by_group=True, n_groups=val_loader.dataset.n_user_groups,
                                  user_to_user_group=val_loader.dataset.user_to_user_group)
        metrics_values = evaluate_recommender_algorithm(alg, val_loader, evaluator,
                                                        verbose=conf['running_settings']['batch_verbose'])
        metrics_values['max_optimizing_metric'] = metrics_values[conf['optimizing_metric']]

        alg.save_model_to_path(conf['model_path'])
        save_yaml(conf['model_path'], conf)

        session.report(metrics_values)

    elif alg in [AlgorithmsEnum.rand, AlgorithmsEnum.pop]:
        warnings.warn("You are running hyperparameter optimization using rand or pop algorithms which do not have any "
                      "hyperparameters. Are you sure of what you are doing?")

        train_dataset = TrainRecDataset(conf['dataset_path'])
        val_loader = get_dataloader(conf, 'val')

        alg = alg.value.build_from_conf(conf, train_dataset)
        evaluator = FullEvaluator(aggr_by_group=True, n_groups=val_loader.dataset.n_user_groups,
                                  user_to_user_group=val_loader.dataset.user_to_user_group)
        metrics_values = evaluate_recommender_algorithm(alg, val_loader, evaluator,
                                                        verbose=conf['running_settings']['batch_verbose'])
        metrics_values['max_optimizing_metric'] = metrics_values[conf['optimizing_metric']]
        save_yaml(conf['model_path'], conf)
        session.report(metrics_values)
    else:
        raise ValueError(f'Training for {alg.value} has been not implemented')

    return metrics_values


def run_train_val(alg: AlgorithmsEnum, dataset: DatasetsEnum, data_path: str, **hyperparameter_settings):
    """
    Runs the train and validation procedure.
    """
    print('Starting Train and Val')

    conf = {
        'data_path': data_path,
        '_in_tune': True,
        'hyperparameter_settings': hyperparameter_settings,
        'device': 'cuda' if hyperparameter_settings['n_gpus'] > 0 else 'cpu',
        **alg_data_param[(alg, dataset)],
    }

    # Adding default parameters if not set
    conf = parse_conf(conf, alg, dataset)
    time_run = conf['time_run']
    seed = conf['running_settings']['seed']

    # Hyperparameter Optimization
    # The actual optimizing metric should be set in conf_parser.py and not here.
    # During hyperparameter optimization we are interested on the maximum of that metric over all the epochs
    # This is used as an early stopping criteria.
    optimizing_metric = 'max_optimizing_metric'

    # Search Algorithm
    search_alg = HyperOptSearch(random_state_seed=seed)

    # Logger
    tags = [alg.name, dataset.name, time_run, 'hyper', f'seed_{seed}']
    if hyperparameter_settings['tags']:
        tags += hyperparameter_settings['tags']

    log_callback = WandbLoggerCallback(project=PROJECT_NAME, log_config=True, api_key_file=WANDB_API_KEY_PATH,
                                       job_type='hyper - train/val',
                                       tags=tags,
                                       entity=ENTITY_NAME, group=f'{alg.name} - {dataset.name} - hyper - train/val')

    # Saving the models only for the best n_tops models.
    keep_callback = KeepOnlyTopModels(optimizing_metric, n_tops=3)

    # Setting up Tune configurations
    tune_config = tune.TuneConfig(
        metric=optimizing_metric,
        mode='max',
        search_alg=search_alg,
        num_samples=hyperparameter_settings["n_samples"],
        max_concurrent_trials=hyperparameter_settings["n_concurrent"],
        trial_name_creator=lambda x: generate_id(postfix=x.trial_id),
        trial_dirname_creator=lambda x: generate_id(postfix=x.trial_id)
    )

    run_config = air.RunConfig(
        storage_path=f'./hyper_saved_models/{alg.name}-{dataset.name}',
        name=time_run,
        callbacks=[log_callback, keep_callback],
        failure_config=air.FailureConfig(max_failures=3),
        verbose=conf['running_settings']['ray_verbose'],
    )

    # Setting up the resources per trial
    dict_resources = {'cpu': hyperparameter_settings['n_cpus']}
    if hyperparameter_settings['n_gpus'] > 0:
        dict_resources['gpu'] = hyperparameter_settings['n_gpus']
    tune_training_with_resources = tune.with_resources(tune_training, dict_resources)

    tuner = tune.Tuner(tune_training_with_resources,
                       param_space=conf,
                       tune_config=tune_config,
                       run_config=run_config)
    results = tuner.fit()

    print('Hyperparameter optimization ended')
    best_result = results.get_best_result()
    best_value = best_result.metrics['max_optimizing_metric']
    best_config = best_result.config
    best_checkpoint = best_result.log_dir

    print(f'Best val value is: \n {best_value}')
    print(f'Best configuration is: \n {best_config}')
    print(f'Best checkpoint is: \n {best_checkpoint}')

    return best_config, best_checkpoint


def run_test(alg: AlgorithmsEnum, dataset: DatasetsEnum, conf: dict, **kwargs):
    """
    Runs the test procedure.
    """
    print('Starting Test')
    time_run = conf['time_run']
    seed = conf['running_settings']['seed']

    tags = [alg.name, dataset.name, time_run, 'hyper', f'seed_{seed}']
    if kwargs['tags']:
        tags += kwargs['tags']
    wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME, config=conf,
               tags=tags,
               group=f'{alg.name} - {dataset.name} - hyper - test', name=time_run, job_type='hyper - test', reinit=True)

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
    metrics_values = evaluate_recommender_algorithm(alg, test_loader, evaluator)
    wandb.log(metrics_values)
    wandb.finish()

    return metrics_values


def start_hyper(alg: AlgorithmsEnum, dataset: DatasetsEnum, data_path: str, **kwargs):
    print('Starting Hyperparameter Optimization')
    print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')

    # ------ Run train and Val ------ #
    best_config, best_checkpoint = run_train_val(alg, dataset, data_path, **kwargs)
    # ------ Run test ------ #
    metric_values = run_test(alg, dataset, best_config, **kwargs)

    return metric_values
