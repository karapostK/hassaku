from hyperopt import hp
from ray import tune

from consts.enums import RecAlgorithmsEnum
from train.rec_losses import RecommenderSystemLossesEnum

base_hyper_params = {
    'train_neg_strategy': tune.choice(['popular', 'uniform']),
    'rec_loss': tune.choice([loss.name for loss in RecommenderSystemLossesEnum]),
    'batch_size': tune.choice([128, 256, 512]),
    'neg_train': tune.randint(1, 20),
    'optim_param': {
        'optim': tune.choice(['adam', 'adagrad']),
        'wd': tune.loguniform(1e-5, 1e-2),
        'lr': tune.loguniform(1e-4, 1e-2)
    },
}

sgdmf_hyper_params = {
    **base_hyper_params,
    'embedding_dim': tune.choice([8, 16, 32, 64, 128, 256, 512]),
    'use_user_bias': tune.choice([True, False]),
    'use_item_bias': tune.choice([True, False]),
    'use_global_bias': tune.choice([True, False])
}

# rbmf_hyper_param = {
#    **base_param,
#    'n_representatives': tune.randint(10, 100),
#    'lam': tune.loguniform(1e-4, 1e-1)
# }

acf_hyper_params = {
    **base_hyper_params,
    'latent_dimension': tune.choice([8, 16, 32, 64, 128, 256, 512]),
    'n_anchors': tune.randint(10, 100),
    'delta_exc': tune.loguniform(1e-3, 10),
    'delta_inc': tune.loguniform(1e-3, 10),
    'loss_aggregator': 'sum'
}

knn_hyper_param = {
    'k': tune.randint(5, 1000),
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
    RecAlgorithmsEnum.pop: {},
    RecAlgorithmsEnum.rand: {},
    RecAlgorithmsEnum.acf: acf_hyper_params
}
