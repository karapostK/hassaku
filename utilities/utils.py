import functools
import logging
import random
from datetime import datetime

import numpy as np
import torch


def generate_id(prefix=None, postfix=None):
    dateTimeObj = datetime.now()
    uid = '{}-{}-{}_{}-{}-{}.{}'.format(dateTimeObj.year, dateTimeObj.month, dateTimeObj.day, dateTimeObj.hour,
                                        dateTimeObj.minute, dateTimeObj.second, dateTimeObj.microsecond)
    if not prefix is None:
        uid = prefix + "_" + uid
    if not postfix is None:
        uid = uid + "_" + postfix
    return uid


def reproducible(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_info_results(metrics):
    """
    Logs the results using logging

    :param metrics: dict containing the metrics to print.
    :return:
    """

    STR_RESULT = "{:10} : {:.5f}"

    for metric_name, metric_value in metrics.items():
        logging.info(STR_RESULT.format(metric_name, metric_value))


class FunctionWrapper:
    """
    Since functions are not properly recognized as enum items, we need to use a wrapper function.
    """

    def __init__(self, function):
        self.function = function
        functools.update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __repr__(self):
        return self.function.__repr__()
