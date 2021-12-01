import json
import pickle
import random
from datetime import datetime

import numpy as np
import torch
from torch import nn


def pickle_load(file_path):
    return pickle.load(open(file_path, 'rb'))


def pickle_dump(x, file_path):
    return pickle.dump(x, open(file_path, 'wb'))


def json_load(file_path):
    return json.load(open(file_path, 'r'))


def general_weight_init(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        if m.weight.requires_grad:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
                torch.nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Embedding:
        if m.weight.requires_grad:
            torch.nn.init.normal_(m.weight)

    elif type(m) == nn.BatchNorm2d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


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


def print_results(metrics):
    """
    Prints the results on the command line.

    :param metrics: dict containing the metrics to print.
    :return:
    """

    STR_RESULT = "{:10} : {:.3f}"

    for metric_name, metric_value in metrics.items():
        print(STR_RESULT.format(metric_name, metric_value))



def generate_slices(total_columns):
    """
    Generate slices that will be processed based on the number of cores
    available on the machine.
    """
    from multiprocessing import cpu_count

    cores = cpu_count()
    print('Running on {} cores'.format(cores))
    segment_length = total_columns // cores

    ranges = []
    now = 0

    while now < total_columns:
        end = now + segment_length

        # The last part can be a little greater that others in some cases, but
        # we can't generate more than #cores ranges
        end = end if end + segment_length <= total_columns else total_columns
        ranges.append((now, end))
        now = end

    return ranges
