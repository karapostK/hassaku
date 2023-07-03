from hyperopt import hp
from ray import tune

from algorithms.algorithms_utils import AlgorithmsEnum
from data.data_utils import DatasetsEnum

knn_hyper_param = {
    'k': tune.randint(5, 100),
    'shrinkage': tune.randint(0, 100),
    'sim_func_params': hp.choice('knn_sim_func_params',
                                 [
                                     {
                                         'sim_func_name': 'jaccard'
                                     },
                                     {
                                         'sim_func_name': 'cosine'
                                     },
                                     # {
                                     #     'sim_func_name': 'sorensen_dice'
                                     # },
                                     # {
                                     #     'sim_func_name': 'asymmetric_cosine',
                                     #     'alpha': hp.uniform('asymmetric_cosine_alpha', 0, 1)
                                     # },
                                     # {
                                     #     'sim_func_name': 'tversky',
                                     #     'alpha': hp.uniform('tversky_alpha', 0, 1),
                                     #     'beta': hp.uniform('tversky_beta', 0, 1)
                                     # }

                                 ])
}
bias_hyper_param = {
    'lr': tune.loguniform(1e-4, 1e-1),
    'wd': tune.loguniform(1e-6, 1e-1),
    # 'train_batch_size': tune.lograndint(32, 4028, base=2),
    # 'rec_loss': tune.choice(['bce', 'bpr', 'sampled_softmax']),
    'neg_train': tune.randint(1, 100),
    # 'train_neg_strategy': tune.choice(['uniform', 'popular'])

}

mf_hyper_param = {
    'lr': tune.loguniform(1e-4, 1e-2),
    'wd': tune.loguniform(1e-6, 1e-2),
    'embedding_dim': tune.randint(10, 256),
    # 'train_batch_size': tune.lograndint(32, 256, base=2),
    'neg_train': tune.randint(1, 100),
    #    'train_neg_strategy': tune.choice(['uniform', 'popular'])
}

als_hyper_param = {
    'factors': tune.randint(10, 256),
    'alpha': tune.loguniform(1e-2, 1),
    'regularization': tune.loguniform(1e-4, 1e-1),
}

protomf_hyper_param = {
    'lr': tune.loguniform(1e-4, 1e-2),
    'wd': tune.loguniform(1e-6, 1e-2),
    'embedding_dim': tune.randint(10, 512),
    'neg_train': tune.randint(1, 100),
    'n_prototypes': tune.randint(10, 50),
    'sim_proto_weight': tune.loguniform(1e-6, 1),
    'sim_batch_weight': tune.loguniform(1e-6, 1)
}

alg_param = {
    AlgorithmsEnum.uknn: knn_hyper_param,
    AlgorithmsEnum.iknn: knn_hyper_param,
    AlgorithmsEnum.mf: mf_hyper_param,
    AlgorithmsEnum.sgdbias: bias_hyper_param,
    AlgorithmsEnum.als: als_hyper_param,
    AlgorithmsEnum.uprotomf: protomf_hyper_param,
    AlgorithmsEnum.iprotomf: protomf_hyper_param
}

# Dataset Specific Parameters

ml1m_common_param = {
    'n_epochs': 50,
    'max_patience': 5,
    'running_settings':
        {
            'n_workers': 0
        },
    'eval_batch_size': 256,
}

lfm2b2020_common_param = {
    'n_epochs': 50,
    'max_patience': 5,
    'running_settings':
        {
            'n_workers': 0
        },
    'eval_batch_size': 8,
}

# KNN

knn_ml1m_param = {
    # Hyper Parameters
    'k': tune.randint(5, 500),
    'shrinkage': tune.randint(0, 500),
    # Set Parameters
    'sim_func_params': {
        'sim_func_name': 'cosine'
    }
}

knn_lfm2b2020_param = {
    # Hyper Parameters
    'k': tune.randint(5, 1000),
    'shrinkage': tune.randint(0, 1000),
    # Set Parameters
    'sim_func_params': {
        'sim_func_name': 'cosine'
    },
    'eval_batch_size': 256
}

# Alternating Least Squares

als_lfm2b2020_param = {
    # Hyper Parameters

    'alpha': tune.randint(20, 100),
    'regularization': tune.loguniform(1e-3, 1),
    # Set Parameters
    'n_iterations': 16,
    'use_gpu': False,
    'eval_batch_size': 40,
    'factors': 2048,
}

# SGD Matrix Factorization

mf_ml1m_param = {
    # Hyper Parameters
    'lr': tune.loguniform(1e-4, 1e-2),
    'wd': tune.loguniform(1e-6, 1e-1),
    'embedding_dim': tune.lograndint(8, 512, base=2),
    'train_batch_size': tune.lograndint(32, 256, base=2),
    'neg_train': tune.randint(1, 100),
    # Set Parameters
    **ml1m_common_param,
    'train_neg_strategy': 'uniform',
    'rec_loss': 'bpr',
    'use_user_bias': False,
    'use_item_bias': True,
    'use_global_bias': False,
    'optimizer': 'adamw'
}

mf_lfm2b2020_param = {
    # Hyper Parameters
    'lr': tune.loguniform(1e-4, 1e-1),
    'wd': tune.loguniform(1e-6, 1e-1),
    'embedding_dim': tune.lograndint(512, 1024, base=2),
    'train_batch_size': tune.lograndint(32, 256, base=2),
    'neg_train': tune.randint(1, 100),
    # Set Parameters
    **lfm2b2020_common_param,
    'train_neg_strategy': 'uniform',
    'rec_loss': 'bpr',
    'use_user_bias': False,
    'use_item_bias': True,
    'use_global_bias': False,
    'optimizer': 'adamw'
}

# ProtoMF

protomf_ml1m_param = {
    # Hyper Parameters
    'lr': tune.loguniform(1e-4, 1e-1),
    'wd': tune.loguniform(1e-6, 1e-2),
    'embedding_dim': tune.lograndint(8, 512, base=2),
    'train_batch_size': tune.lograndint(32, 256, base=2),
    'neg_train': tune.randint(1, 100),
    'n_prototypes': tune.randint(5, 100),
    'sim_batch_weight': 0,  # tune.loguniform(1e-5, 1e-1),
    'sim_proto_weight': 0,  # tune.loguniform(1e-6, 1e-3),
    # Set Parameters
    **ml1m_common_param,
    'train_neg_strategy': 'uniform',
    'rec_loss': 'sampled_softmax',
    'optimizer': 'adamw'
}

protomf_lfm2b2020_param = {
    # Hyper Parameters
    'lr': tune.loguniform(1e-4, 1e-1),
    'wd': tune.loguniform(1e-6, 1e-1),
    'embedding_dim': tune.lograndint(8, 512, base=2),
    'train_batch_size': tune.lograndint(32, 256, base=2),
    'neg_train': tune.randint(1, 100),
    'n_prototypes': tune.randint(5, 50),
    'sim_proto_weight': tune.loguniform(1e-6, 1),
    'sim_batch_weight': tune.loguniform(1e-6, 1),
    # Set Parameters
    **lfm2b2020_common_param,
    'train_neg_strategy': 'uniform',
    'rec_loss': 'sampled_softmax',
    'optimizer': 'adamw'
}

alg_data_param = {
    (AlgorithmsEnum.mf, DatasetsEnum.ml1m): mf_ml1m_param,
    (AlgorithmsEnum.mf, DatasetsEnum.lfm2b2020): mf_lfm2b2020_param,
    (AlgorithmsEnum.uprotomf, DatasetsEnum.ml1m): protomf_ml1m_param,
    (AlgorithmsEnum.uprotomf, DatasetsEnum.lfm2b2020): protomf_lfm2b2020_param,
    (AlgorithmsEnum.uknn, DatasetsEnum.ml1m): knn_ml1m_param,
    (AlgorithmsEnum.iknn, DatasetsEnum.ml1m): knn_ml1m_param,
    (AlgorithmsEnum.uknn, DatasetsEnum.lfm2b2020): knn_lfm2b2020_param,
    (AlgorithmsEnum.iknn, DatasetsEnum.lfm2b2020): knn_lfm2b2020_param,
    (AlgorithmsEnum.als, DatasetsEnum.lfm2b2020): als_lfm2b2020_param,
    (AlgorithmsEnum.iprotomf, DatasetsEnum.ml1m): protomf_ml1m_param,
    (AlgorithmsEnum.iprotomf, DatasetsEnum.lfm2b2020): protomf_lfm2b2020_param,
    (AlgorithmsEnum.uprotomf2, DatasetsEnum.ml1m): protomf_ml1m_param
}
