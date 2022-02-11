import enum
from enum import Enum

from algorithms.knn_algs import UserKNN, ItemKNN
from algorithms.linear_algs import SLIM
from algorithms.mf_algs import SVDAlgorithm, AlternatingLeastSquare, RBMF
from algorithms.naive_algs import PopularItems, RandomItems
from algorithms.neural_alg import SGDMatrixFactorization, UProtoMF, UIProtoMF


class RecAlgorithmsEnum(Enum):
    svd = SVDAlgorithm
    uknn = UserKNN
    iknn = ItemKNN
    slim = SLIM
    sgdmf = SGDMatrixFactorization
    als = AlternatingLeastSquare
    rbmf = RBMF
    uprotomf = UProtoMF
    uiprotomf = UIProtoMF
    pop = PopularItems
    rand = RandomItems


class RecDatasetsEnum(Enum):
    ml1m = enum.auto()
    amazon2014 = enum.auto()
    lfm2b1m = enum.auto()
