from ray import tune

from algorithms.algorithms_utils import AlgorithmsEnum
from data.data_utils import DatasetsEnum

# Default parameters

N_EPOCHS = 50
MAX_PATIENCE = 5

# Dataset Specific Parameters
all_dataset_common_param = {
    'n_epochs': N_EPOCHS,
    'max_patience': MAX_PATIENCE,
    'running_settings':
        {
            'n_workers': 0
        },
}

# K-Nearest Neighbour (KNN)

knn_param = {
    # Hyper Parameters
    'k': tune.randint(3, 100),
    'shrinkage': tune.randint(0, 500),
    # Set Parameters
    'sim_func_params': {
        'sim_func_name': 'cosine'
    },
    'eval_batch_size': 128
}

# Representative Based Matrix Factorization (RBMF)

rbmf_param = {
    # Hyper Parameters
    'n_representatives': tune.randint(5, 100),
    'lam': tune.loguniform(1e-6, 1),
    # Set Parameter
    'eval_batch_size': 256
}

# Alternating Least Squares (ALS)

als_param = {
    # Hyper Parameters
    'alpha': tune.randint(20, 100),
    'regularization': tune.loguniform(1e-3, 1),
    'factors': tune.lograndint(8, 512),
    # Set Parameters
    'n_iterations': 16,
    'use_gpu': False,
    'eval_batch_size': 40,

}

# Stochastic Gradient Descent -based Matrix Factorization (MF)

mf_param = {
    # Hyper Parameters
    'lr': tune.loguniform(1e-4, 1e-2),
    'wd': tune.loguniform(1e-6, 1e-1),
    'embedding_dim': tune.lograndint(8, 512, base=2),
    'train_batch_size': tune.lograndint(32, 256, base=2),
    'neg_train': tune.randint(1, 100),
    # Set Parameters
    **all_dataset_common_param,
    'train_neg_strategy': 'uniform',
    'rec_loss': 'bpr',
    'use_user_bias': False,
    'use_item_bias': True,
    'use_global_bias': False,
    'optimizer': 'adamw'
}

mf_ml1m_param = {
    **mf_param,
    'eval_batch_size': 256,
}

mf_lfm2b2020_param = {
    **mf_param,
    'eval_batch_size': 8,
}

mf_amazonvid2018_param = {
    **mf_param,
    'eval_batch_size': 64,
}

# ACF

acf_param = {
    # Hyper Parameters
    'lr': tune.loguniform(1e-4, 1e-1),
    'wd': tune.loguniform(1e-5, 1e-1),
    'embedding_dim': tune.lograndint(8, 512, base=2),
    'train_batch_size': tune.lograndint(32, 256, base=2),
    'neg_train': tune.randint(1, 100),
    'delta_exc': tune.loguniform(1e-6, 1e-2),
    'delta_inc': tune.loguniform(1e-6, 1e2),
    'n_anchors': tune.randint(5, 100),
    # Set Parameters
    **all_dataset_common_param,
    'train_neg_strategy': 'uniform',
    'rec_loss': 'sampled_softmax',
    'optimizer': 'adamw',
}

acf_ml1m_param = {
    **acf_param,
    'eval_batch_size': 256,
}

acf_lfm2b2020_param = {
    **acf_param,
    'eval_batch_size': 16,
}

acf_amazonvid2018_param = {
    **acf_param,
    'eval_batch_size': 64,
}

# ProtoMF

protomf_param = {
    # Hyper Parameters
    'lr': tune.loguniform(1e-4, 1e-2),
    'wd': tune.loguniform(1e-6, 1e-2),
    'embedding_dim': tune.lograndint(8, 512, base=2),
    'train_batch_size': tune.lograndint(32, 256, base=2),
    'neg_train': tune.randint(1, 100),
    'n_prototypes': tune.randint(5, 100),
    'sim_batch_weight': tune.loguniform(1e-2, 1),
    'sim_proto_weight': tune.loguniform(1e-2, 1),
    # Set Parameters
    **all_dataset_common_param,
    'train_neg_strategy': 'uniform',
    'rec_loss': 'sampled_softmax',
    'optimizer': 'adamw'
}

protomf_ml1m_param = {
    **protomf_param,
    'eval_batch_size': 256,
}

protomf_lfm2b2020_param = {
    **protomf_param,
    'eval_batch_size': 16,
}

protomf_amazonvid2018_param = {
    **protomf_param,
    'eval_batch_size': 64,
}

# UIProtoMF

uiprotomf_param = {
    # Hyper Parameters
    'lr': tune.loguniform(1e-4, 1e-2),
    'wd': tune.loguniform(1e-6, 1e-2),
    'embedding_dim': tune.lograndint(8, 512, base=2),
    'train_batch_size': tune.lograndint(32, 256, base=2),
    'neg_train': tune.randint(1, 100),
    'u_n_prototypes': tune.randint(5, 50),
    'i_n_prototypes': tune.randint(5, 50),
    'u_sim_batch_weight': tune.loguniform(1e-2, 1),
    'u_sim_proto_weight': tune.loguniform(1e-2, 1),
    'i_sim_batch_weight': tune.loguniform(1e-2, 1),
    'i_sim_proto_weight': tune.loguniform(1e-2, 1),
    # Set Parameters
    **all_dataset_common_param,
    'train_neg_strategy': 'uniform',
    'rec_loss': 'sampled_softmax',
    'optimizer': 'adamw'
}

uiprotomf_ml1m_param = {
    **uiprotomf_param,
    'eval_batch_size': 256,
}

uiprotomf_lfm2b2020_param = {
    **uiprotomf_param,
    'eval_batch_size': 16,
}

uiprotomf_amazonvid2018_param = {
    **uiprotomf_param,
    'eval_batch_size': 64,
}

alg_data_param = {
    # Sharing KNN parameters across uknn/iknn and across datasets.
    (AlgorithmsEnum.uknn, DatasetsEnum.ml1m): knn_param,
    (AlgorithmsEnum.iknn, DatasetsEnum.ml1m): knn_param,
    (AlgorithmsEnum.uknn, DatasetsEnum.lfm2b2020): knn_param,
    (AlgorithmsEnum.iknn, DatasetsEnum.lfm2b2020): knn_param,
    (AlgorithmsEnum.uknn, DatasetsEnum.amazonvid2018): knn_param,
    (AlgorithmsEnum.iknn, DatasetsEnum.amazonvid2018): knn_param,
    # Sharing RBMF parameters across datasets
    (AlgorithmsEnum.rbmf, DatasetsEnum.ml1m): rbmf_param,
    (AlgorithmsEnum.rbmf, DatasetsEnum.lfm2b2020): rbmf_param,
    (AlgorithmsEnum.rbmf, DatasetsEnum.amazonvid2018): rbmf_param,
    # Sharing ALS parameters across datasets
    (AlgorithmsEnum.als, DatasetsEnum.ml1m): als_param,
    (AlgorithmsEnum.als, DatasetsEnum.lfm2b2020): als_param,
    (AlgorithmsEnum.als, DatasetsEnum.amazonvid2018): als_param,
    # ACF
    (AlgorithmsEnum.acf, DatasetsEnum.ml1m): acf_ml1m_param,
    (AlgorithmsEnum.acf, DatasetsEnum.lfm2b2020): acf_lfm2b2020_param,
    (AlgorithmsEnum.acf, DatasetsEnum.amazonvid2018): acf_amazonvid2018_param,
    # Matrix Factorization
    (AlgorithmsEnum.mf, DatasetsEnum.ml1m): mf_ml1m_param,
    (AlgorithmsEnum.mf, DatasetsEnum.lfm2b2020): mf_lfm2b2020_param,
    (AlgorithmsEnum.mf, DatasetsEnum.amazonvid2018): mf_amazonvid2018_param,
    # ProtoMF
    (AlgorithmsEnum.uprotomf, DatasetsEnum.ml1m): protomf_ml1m_param,
    (AlgorithmsEnum.uprotomf, DatasetsEnum.lfm2b2020): protomf_lfm2b2020_param,
    (AlgorithmsEnum.uprotomf, DatasetsEnum.amazonvid2018): protomf_amazonvid2018_param,
    (AlgorithmsEnum.iprotomf, DatasetsEnum.ml1m): protomf_ml1m_param,
    (AlgorithmsEnum.iprotomf, DatasetsEnum.lfm2b2020): protomf_lfm2b2020_param,
    (AlgorithmsEnum.iprotomf, DatasetsEnum.amazonvid2018): protomf_amazonvid2018_param,
    # UIProtoMF
    (AlgorithmsEnum.uiprotomf, DatasetsEnum.ml1m): uiprotomf_ml1m_param,
    (AlgorithmsEnum.uiprotomf, DatasetsEnum.lfm2b2020): uiprotomf_lfm2b2020_param,
    (AlgorithmsEnum.uiprotomf, DatasetsEnum.amazonvid2018): uiprotomf_amazonvid2018_param,
}
