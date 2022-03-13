import torch
from hyperopt import hp
from ray import tune

from consts.enums import RecAlgorithmsEnum
from train.rec_losses import RecommenderSystemLossesEnum

base_param = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'eval_batch_size': 256,
}

base_hyper_params = {
    **base_param,
    'train_neg_strategy': tune.choice(['popular', 'uniform']),
    'rec_loss': tune.choice([RecommenderSystemLossesEnum.bce, RecommenderSystemLossesEnum.bpr,
                             RecommenderSystemLossesEnum.sampled_softmax]),
    'batch_size': tune.lograndint(64, 512, 2),  # todo: maybe drop it
    'optim_param': {
        'optim': tune.choice(['adam', 'adagrad']),
        'wd': tune.loguniform(1e-4, 1e-2),
        'lr': tune.loguniform(1e-4, 1e-1)
    },
}

sgdmf_hyper_params = {
    **base_hyper_params,
    'embedding_dim': tune.randint(10, 100),
    'use_user_bias': tune.choice([True, False]),
    'use_item_bias': tune.choice([True, False]),
    'use_global_bias': tune.choice([True, False])
}

# rbmf_hyper_param = {
#    **base_param,
#    'n_representatives': tune.randint(10, 100),
#    'lam': tune.loguniform(1e-4, 1e-1)
# }

knn_hyper_param = {
    **base_param,
    'k': tune.randint(1, 100),
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

protomf_hyper_param = {
    **base_hyper_params,
    'latent_dimension': tune.randint(10, 100),
    'n_prototypes': tune.randint(10, 100),
    'sim_proto_weight': tune.loguniform(1e-3, 10),
    'sim_batch_weight': tune.loguniform(1e-3, 10)
}

uiprotomf_hyper_param = {
    **base_hyper_params,
    'latent_dimension': tune.randint(10, 100),
    'u_n_prototypes': tune.randint(10, 100),
    'i_n_prototypes': tune.randint(10, 100),
    'u_sim_proto_weight': tune.loguniform(1e-3, 10),
    'u_sim_batch_weight': tune.loguniform(1e-3, 10),
    'i_sim_proto_weight': tune.loguniform(1e-3, 10),
    'i_sim_batch_weight': tune.loguniform(1e-3, 10),

}
alg_param = {
    RecAlgorithmsEnum.sgdbias: base_hyper_params,
    RecAlgorithmsEnum.uknn: knn_hyper_param,
    RecAlgorithmsEnum.iknn: knn_hyper_param,
    RecAlgorithmsEnum.sgdmf: sgdmf_hyper_params,
    RecAlgorithmsEnum.uprotomf: protomf_hyper_param,
    RecAlgorithmsEnum.iprotomf: protomf_hyper_param,
    RecAlgorithmsEnum.uiprotomf: uiprotomf_hyper_param,
}
