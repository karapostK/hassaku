from algorithms.algorithms_utils import AlgorithmsEnum
from data.data_utils import DatasetsEnum
from experiment_helper import run_train_val_test

for dataset in DatasetsEnum:
    for alg in [AlgorithmsEnum.rand, AlgorithmsEnum.pop, AlgorithmsEnum.uknn, AlgorithmsEnum.iknn]:
        if alg in [AlgorithmsEnum.rand, AlgorithmsEnum.pop]:
            conf = './conf/naive_conf.yml'
        elif alg in [AlgorithmsEnum.uknn, AlgorithmsEnum.iknn]:
            conf = './conf/knn_conf.yml'
        else:
            raise ValueError()

        run_train_val_test(alg, dataset, conf)
