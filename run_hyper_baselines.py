from algorithms.algorithms_utils import AlgorithmsEnum
from data.data_utils import DatasetsEnum
from hyper_search.experiment_helper import start_hyper

conf = './conf/knn_conf.yml'
for dataset in DatasetsEnum:
    if dataset == DatasetsEnum.ml1m:
        continue
    for alg in [AlgorithmsEnum.uknn, AlgorithmsEnum.iknn]:
        start_hyper(alg, dataset, conf, n_gpus=0.2, n_concurrent=4, n_samples=50)
