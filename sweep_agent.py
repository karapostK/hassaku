import wandb
from algorithms.algorithms_utils import AlgorithmsEnum
from algorithms.base_classes import SGDBasedRecommenderAlgorithm, SparseMatrixBasedRecommenderAlgorithm
from conf.conf_parser import parse_conf, save_yaml
from data.data_utils import DatasetsEnum, get_dataloader
from data.dataset import TrainRecDataset
from eval.eval import evaluate_recommender_algorithm
from train.trainer import Trainer
from utilities.utils import reproducible


def train_val_agent():
    # Initialization and gathering hyperparameters
    run = wandb.init(job_type='train/val')

    conf = wandb.config

    alg = AlgorithmsEnum[conf.alg]
    dataset = DatasetsEnum[conf.dataset]

    conf = parse_conf(conf, alg, dataset)

    # Updating wandb data
    run.tags += (alg.name, dataset.name)
    wandb.config.update(conf)

    print('Starting Train-Val')
    print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')

    reproducible(conf['running_settings']['seed'])

    if issubclass(alg.value, SGDBasedRecommenderAlgorithm):

        train_loader = get_dataloader(conf, 'train')
        val_loader = get_dataloader(conf, 'val')

        alg = alg.value.build_from_conf(conf, train_loader.dataset)

        # Validation happens within the Trainer
        trainer = Trainer(alg, train_loader, val_loader, conf)
        trainer.fit()

    elif issubclass(alg.value, SparseMatrixBasedRecommenderAlgorithm):
        train_dataset = TrainRecDataset(conf['dataset_path'])
        val_loader = get_dataloader(conf, 'val')

        alg = alg.value.build_from_conf(conf, train_dataset)
        # -- Training --
        alg.fit(train_dataset.sampling_matrix)
        # -- Validation --
        metrics_values = evaluate_recommender_algorithm(alg, val_loader,
                                                        verbose=conf['running_settings']['batch_verbose'])

        alg.save_model_to_path(conf['model_path'])
        wandb.log(metrics_values)
    elif alg in [AlgorithmsEnum.rand, AlgorithmsEnum.pop]:

        train_dataset = TrainRecDataset(conf['dataset_path'])
        val_loader = get_dataloader(conf, 'val')

        alg = alg.value.build_from_conf(conf, train_dataset)
        metrics_values = evaluate_recommender_algorithm(alg, val_loader,
                                                        verbose=conf['running_settings']['batch_verbose'])
        wandb.log(metrics_values)
    else:
        raise ValueError(f'Training for {alg.value} has been not implemented')
    save_yaml(conf['model_path'], conf)
    wandb.finish()


train_val_agent()
