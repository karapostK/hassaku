import enum
from enum import Enum

from algorithms.knn_algs import UserKNN, ItemKNN
from algorithms.linear_algs import SLIM
from algorithms.mf_algs import SVDAlgorithm, AlternatingLeastSquare
from algorithms.naive_algs import RandomItems, PopularItems
from algorithms.neural_alg import SGDMatrixFactorization


class RecAlgorithmsEnum(Enum):
    random = RandomItems
    popular = PopularItems
    svd = SVDAlgorithm
    uknn = UserKNN
    iknn = ItemKNN
    slim = SLIM
    sgdmf = SGDMatrixFactorization
    als = AlternatingLeastSquare


class RecDatasetsEnum(Enum):
    ml1m = enum.auto()
    amazon2014 = enum.auto()
    lfm2b1m = enum.auto()
