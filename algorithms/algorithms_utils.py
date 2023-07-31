from enum import Enum

from algorithms.graph_algs import P3alpha
from algorithms.knn_algs import UserKNN, ItemKNN
from algorithms.linear_algs import EASE, SLIM
from algorithms.mf_algs import SVDAlgorithm, AlternatingLeastSquare, RBMF
from algorithms.naive_algs import PopularItems, RandomItems
from algorithms.sgd_alg import SGDMatrixFactorization, UProtoMF, UIProtoMF, IProtoMF, SGDBaseline, ACF, UProtoMFs, \
    IProtoMFs, UIProtoMFs


class AlgorithmsEnum(Enum):
    uknn = UserKNN
    iknn = ItemKNN
    mf = SGDMatrixFactorization
    sgdbias = SGDBaseline
    pop = PopularItems
    rand = RandomItems
    rbmf = RBMF
    uprotomf = UProtoMF
    iprotomf = IProtoMF
    uiprotomf = UIProtoMF
    acf = ACF
    svd = SVDAlgorithm
    als = AlternatingLeastSquare
    p3alpha = P3alpha
    ease = EASE
    slim = SLIM
    uprotomfs = UProtoMFs
    iprotomfs = IProtoMFs
    uiprotomfs = UIProtoMFs


