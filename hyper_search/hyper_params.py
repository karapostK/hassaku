from hyperopt import hp
from ray import tune

from algorithms.algorithms_utils import AlgorithmsEnum

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

mf_hyper_param = {
    'embedding_dim': tune.randint(10, 100),
    'lr': tune.loguniform(1e-4, 1e-2),
    'wd': tune.loguniform(1e-6, 1e-2),
    'train_batch_size': tune.lograndint(32, 4028, base=2),
    'rec_loss': tune.choice(['bce', 'bpr', 'sampled_softmax']),
}

bias_hyper_param = {
    'lr': tune.loguniform(1e-4, 1e-1),
    'wd': tune.loguniform(1e-6, 1e-1),
    'train_batch_size': tune.lograndint(32, 4028, base=2),
    'rec_loss': tune.choice(['bce', 'bpr', 'sampled_softmax']),
    'neg_train': tune.randint(1, 100),
    'train_neg_strategy': tune.choice(['uniform', 'popular'])

}
alg_param = {
    AlgorithmsEnum.uknn: knn_hyper_param,
    AlgorithmsEnum.iknn: knn_hyper_param,
    AlgorithmsEnum.mf: mf_hyper_param,
    AlgorithmsEnum.sgdbias: bias_hyper_param
}
