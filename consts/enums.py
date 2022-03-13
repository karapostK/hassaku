import enum
from enum import Enum

from algorithms.knn_algs import UserKNN, ItemKNN
from algorithms.sgd_alg import SGDMatrixFactorization, UProtoMF, UIProtoMF, IProtoMF, SGDBaseline


# from algorithms.mf_algs import RBMF


class RecAlgorithmsEnum(Enum):
    uknn = UserKNN
    iknn = ItemKNN
    sgdmf = SGDMatrixFactorization
    sgdbias = SGDBaseline
    # rbmf = RBMF
    uprotomf = UProtoMF
    iprotomf = IProtoMF
    uiprotomf = UIProtoMF


class RecDatasetsEnum(Enum):
    ml1m = enum.auto()
    # amazon2014 = enum.auto()
    # lfm2b1m = enum.auto()
