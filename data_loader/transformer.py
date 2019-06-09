import torch
import numpy as np
import scipy.signal as signal
import numpy.random as random


class ToTensor(object):

    def __call__(self, sample):
        if len(sample) == 2:
            x, y = torch.from_numpy(sample[0]), torch.from_numpy(sample[1])
            x, y = x.type(torch.FloatTensor), y.type(torch.LongTensor)
            return x, y
        else:
            sample = torch.from_numpy(sample)
            sample = sample.type(torch.FloatTensor)
            return sample


class TripleChanne1(object):

    def __call__(self, sample):
        if len(sample) == 2:
            x = np.repeat(sample[0], repeats=3, axis=0)
            return x, sample[1]
        else:
            return np.repeat(sample, repeats=3, axis=0)


class MedfilterChannel(object):
    """
    medfilter selected channel of triple channel input
    """
    def __init__(self, width=21, height=7, channel_idx=0):
        self.width = width
        self.height = height
        self.channel_idx = channel_idx

    def __call__(self, sample):
        """
        :param sample: ((channel=1 or 3, Height, width), label)
        :return:
        """
        if len(sample) == 2:
            sample[0][self.channel_idx] = self.filter(data=sample[0][self.channel_idx])
        else:
            sample[self.channel_idx] = self.filter(data=sample[self.channel_idx])
        return sample

    def filter(self, data):
        """
        :param data:  2d array
        :return:
        """
        tmp = signal.medfilt2d(input=data, kernel_size=(self.height, self.width))
        data = data - tmp
        return data


class MeanSubtractionChannel(object):
    def __init__(self, channel_idx=0):
        self.channel_idx = channel_idx

    def __call__(self, sample):
        """
        :param sample: ((channel=1 or 3, Height, width), label)
        :return:
        """
        if len(sample) == 2:
            sample[0][self.channel_idx] = sample[0][self.channel_idx] - np.mean(sample[0][self.channel_idx],
                                                                                axis=1, keepdims=True)
        else:
            sample[self.channel_idx] = sample[self.channel_idx] - np.mean(sample[self.channel_idx],
                                                                          axis=1, keepdims=True)
        return sample


class RandomSPL(object):
    """
    random enhance sound pressure level on spectrogram
    """
    def __init__(self, spl_var=1, prob=0.5):
        """
        :param spl_range: a tuple / list, default to (0,0), i.e. no spl perturbation
        """
        self.prob = prob
        self.enhance_number = random.randn() * spl_var
        self.enhance_number = np.clip(self.enhance_number, - 2 * spl_var, 2 * spl_var)

    def __call__(self, sample):

        if np.random.random() < self.prob:
            data = sample[0] + self.enhance_number

            return data, sample[1]
        else:
            return sample
