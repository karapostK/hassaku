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
    'embedding_dim': tune.randint(10, 30)
}
alg_param = {
    AlgorithmsEnum.uknn: knn_hyper_param,
    AlgorithmsEnum.iknn: knn_hyper_param,
    AlgorithmsEnum.mf: mf_hyper_param
}
