import enum
from enum import Enum

from algorithms.naive_algs import RandomItems, PopularItems
from algorithms.mf_algs import SVDAlgorithm


class RecAlgorithmsEnum(Enum):
    random = RandomItems
    popular = PopularItems
    svd = SVDAlgorithm


class RecDatasetsEnum(Enum):
    ml1m = enum.auto()
    amazon2014 = enum.auto()
    lfm2b1m = enum.auto()
