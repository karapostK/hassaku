import enum
from enum import Enum

from algorithms.knn_algs import UserKNN, ItemKNN
from algorithms.naive_algs import PopularItems, RandomItems
from algorithms.sgd_alg import SGDMatrixFactorization, UProtoMF, UIProtoMF, IProtoMF, SGDBaseline, ACF


# from algorithms.mf_algs import RBMF


class RecAlgorithmsEnum(Enum):
    uknn = UserKNN
    iknn = ItemKNN
    # bprmf = SGDMatrixFactorization  # BPR applied to Matrix Factorization
    logmf = SGDMatrixFactorization  # Logistic Matrix Factorization
    sgdbias = SGDBaseline
    pop = PopularItems
    rand = RandomItems
    # rbmf = RBMF
    uprotomf = UProtoMF
    iprotomf = IProtoMF
    uiprotomf = UIProtoMF
    acf = ACF


class RecDatasetsEnum(Enum):
    ml1m = enum.auto()
    amazonvid = enum.auto()
    lfm2b2020 = enum.auto()
