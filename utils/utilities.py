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


def calculate_confusion_matrix(target, predict, classes_num):
    """Calculate confusion matrix.
    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
      classes_num: int, number of classes
    Outputs:
      confusion_matrix: (classes_num, classes_num)
    """

    confusion_matrix = np.zeros((classes_num, classes_num))
    samples_num = len(target)

    for n in range(samples_num):
        confusion_matrix[target[n], predict[n]] += 1

    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, title, labels, values):
    """Plot confusion matrix.
    Inputs:
      confusion_matrix: matrix, (classes_num, classes_num)
      labels: list of labels
      values: list of values to be shown in diagonal
    Ouputs:
      None
    """
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

    if labels:
        ax.set_xticklabels([''] + labels, rotation=90, ha='left')
        ax.set_yticklabels([''] + labels)
        ax.xaxis.set_ticks_position('bottom')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    for n in range(len(values)):
        plt.text(n - 0.4, n, '{:.2f}'.format(values[n]), color='yellow')

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.tight_layout()
    plt.show()


def calculate_accuracy(target, predict, classes_num, average=None):
    """Calculate accuracy.
    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
    Outputs:
      accuracy: float
    """

    samples_num = len(target)

    correctness = np.zeros(classes_num)
    total = np.zeros(classes_num)

    for n in range(samples_num):

        total[target[n]] += 1

        if target[n] == predict[n]:
            correctness[target[n]] += 1

    accuracy = correctness / total

    if average is None:
        return accuracy

    elif average == 'macro':
        return np.mean(accuracy)

    else:
        raise Exception('Incorrect average!')
