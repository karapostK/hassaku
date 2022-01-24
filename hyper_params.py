import torch
from hyperopt import hp
from ray import tune

from utilities.enums import RecAlgorithmsEnum
from utilities.rec_losses import RecommenderSystemLossesEnum

base_param = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_epochs': 100,
    'eval_neg_strategy': 'uniform',
    'val_batch_size': 256,
}

base_hyper_params = {
    **base_param,
    'neg_train': tune.randint(1, 50),
    'train_neg_strategy': tune.choice(['popular', 'uniform']),
    'rec_loss': tune.choice([RecommenderSystemLossesEnum.bce, RecommenderSystemLossesEnum.bpr,
                             RecommenderSystemLossesEnum.sampled_softmax]),
    'batch_size': tune.lograndint(64, 512, 2),
    'optim_param': {
        'optim': tune.choice(['adam', 'adagrad']),
        'wd': tune.loguniform(1e-4, 1e-2),
        'lr': tune.loguniform(1e-4, 1e-1)
    },
}

sgdmf_hyper_params = {
    **base_hyper_params,
    'embedding_dim': tune.randint(10, 100)
}

svd_hyper_param = {
    **base_param,
    'n_factors': tune.randint(10, 100),
}

rbmf_hyper_param = {
    **base_param,
    'n_representatives': tune.randint(10, 100),
    'lam': tune.loguniform(1e-4, 1e-1)
}
knn_hyper_param = {
    **base_param,
    'k': tune.lograndint(1, 1000),
    'sim_func_params': hp.choice('sim_func_name', [
        {
            'sim_func_name': 'jaccard'
        },
        {
            'sim_func_name': 'cosine'
        },
        {
            'sim_func_name': 'sorensen_dice'
        },
        {
            'sim_func_name': 'asymmetric_cosine',
            'alpha': hp.uniform('asymmetric_cosine_alpha', 0, 1)
        },
        {
            'sim_func_name': 'tversky',
            'alpha': hp.uniform('tversky_alpha', 0, 1),
            'beta': hp.uniform('tversky_beta', 0, 1)
        }

    ])
}

slim_hyper_param = {
    **base_param,
    'alpha': tune.loguniform(1e-4, 100),
    'l1_ratio': tune.loguniform(1e-4, 1),
    'max_iter': tune.randint(100, 500)
}

als_hyper_param = {
    **base_param,
    'alpha': tune.randint(1, 100),
    'factors': tune.randint(10, 100),
    'regularization': tune.loguniform(1e-4, 1e-1),
    'n_iterations': tune.randint(100, 1000)
}

alg_param = {
    RecAlgorithmsEnum.random: base_param,
    RecAlgorithmsEnum.popular: base_param,
    RecAlgorithmsEnum.svd: svd_hyper_param,
    RecAlgorithmsEnum.uknn: knn_hyper_param,
    RecAlgorithmsEnum.iknn: knn_hyper_param,
    RecAlgorithmsEnum.slim: slim_hyper_param,
    RecAlgorithmsEnum.sgdmf: sgdmf_hyper_params,
    RecAlgorithmsEnum.als: als_hyper_param,
    RecAlgorithmsEnum.rbmf: rbmf_hyper_param
}
