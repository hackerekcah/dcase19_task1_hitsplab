import numpy as np
import numpy.random as random
import math
import torch
import logging
import os


def standarize_with_norm(data_list, norm_data):
    """
    :param data_list: given a list of data to normalize, input data shape [batch, frequency, time]
    :param norm_data: given a matrix for calc mean and variance, norm data shape [batch, frequency, time]
    :return:
    """
    # transpose to (batch, time, frequency)
    norm_data = np.transpose(norm_data, [0, 2, 1])
    # mean, std
    mu = np.mean(norm_data, axis=(0, 1))
    sigma = np.std(norm_data, axis=(0, 1))

    scaled = []
    for data in data_list:
        data = np.transpose(data, [0, 2, 1])
        data = (data - mu) / sigma
        # transpose back to (batch, frequency, time)
        data = np.transpose(data, [0, 2, 1])
        scaled.append(data)

    return scaled


def mean_sub_norm(data_list, norm_data):
    # transpose to (batch, time, frequency)
    norm_data = np.transpose(norm_data, [0, 2, 1])
    # mean, std
    mu = np.mean(norm_data, axis=(0, 1))
    # sigma = np.std(norm_data, axis=(0, 1))

    scaled = []
    for data in data_list:
        data = np.transpose(data, [0, 2, 1])
        data = data - mu
        # transpose back to (batch, frequency, time)
        data = np.transpose(data, [0, 2, 1])
        scaled.append(data)

    return scaled


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def set_logging(root_dir, args):

    # setup logging info
    log_file = '{}/ckpt/{}/{}.log'.format(root_dir, args.exp, args.ckpt_prefix)
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    fileh = logging.FileHandler(log_file, 'a')
    fileh.setLevel('INFO')

    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    log.setLevel('INFO')
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)  # set the new handler
