from torch.utils.data import DataLoader
from data_loader.transformer import *
from torchvision.transforms import Compose
from data_loader.data_sets import *


class LB_Loader:
    def __init__(self, is_divide_variance=True):
        self.is_divide_variance = is_divide_variance

    def train(self, batch_size=128, shuffle=True):
        return DataLoader(dataset=TrainSet(is_divide_variance=self.is_divide_variance,
                                           transform=Compose([TripleChanne1(), ToTensor()])),
                          batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=3)

    def val(self, batch_size=128):
        return DataLoader(dataset=ValSet(is_divide_variance=self.is_divide_variance,
                                         transform=Compose([TripleChanne1(), ToTensor()])),
                          batch_size=batch_size, shuffle=False, num_workers=3)

    def dev(self, batch_size=128, shuffle=True):
        return DataLoader(dataset=DevDataSet(is_divide_variance=self.is_divide_variance,
                                             transform=Compose([TripleChanne1(), ToTensor()])),
                          batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=3)

    def lb(self, batch_size=128):
        return DataLoader(dataset=LB_Dataset(is_divide_variance=self.is_divide_variance,
                                             transform=Compose([TripleChanne1(),
                                                                ToTensor()])),
                          batch_size=batch_size, shuffle=False, num_workers=3)


class LB_SPL_Loader:
    def __init__(self, is_divide_variance=True):
        self.is_divide_variance = is_divide_variance

    def train(self, batch_size=128, shuffle=True):
        return DataLoader(dataset=TrainSet(is_divide_variance=self.is_divide_variance,
                                           transform=Compose([RandomSPL(spl_var=1.0, prob=0.5),
                                                              TripleChanne1(),
                                                              ToTensor()])),
                          batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=3)

    def val(self, batch_size=128):
        return DataLoader(dataset=ValSet(is_divide_variance=self.is_divide_variance,
                                         transform=Compose([TripleChanne1(), ToTensor()])),
                          batch_size=batch_size, shuffle=False, num_workers=3)

    def dev(self, batch_size=128, shuffle=True):
        return DataLoader(dataset=DevDataSet(is_divide_variance=self.is_divide_variance,
                                             transform=Compose([RandomSPL(spl_var=1.0, prob=0.5),
                                                                TripleChanne1(),
                                                                ToTensor()])),
                          batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=3)

    def lb(self, batch_size=128):
        return DataLoader(dataset=LB_Dataset(is_divide_variance=self.is_divide_variance,
                                             transform=Compose([TripleChanne1(),
                                                                ToTensor()
                                                                ])),
                          batch_size=batch_size, shuffle=False, num_workers=3)


class LB_Medfilter_Loader:
    def __init__(self, is_divide_variance=True):
        self.is_divide_variance = is_divide_variance

    def train(self, batch_size=128, shuffle=True):
        return DataLoader(dataset=TrainSet(is_divide_variance=self.is_divide_variance,
                                           transform=Compose([MedfilterChannel(width=21, height=7, channel_idx=0),
                                                              TripleChanne1(),
                                                              ToTensor()])),
                          batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=3)

    def val(self, batch_size=128):
        return DataLoader(dataset=ValSet(is_divide_variance=self.is_divide_variance,
                                         transform=Compose([MedfilterChannel(width=21, height=7, channel_idx=0),
                                                            TripleChanne1(),
                                                            ToTensor()])),
                          batch_size=batch_size, shuffle=False, num_workers=3)

    def dev(self, batch_size=128, shuffle=True):
        return DataLoader(dataset=DevDataSet(is_divide_variance=self.is_divide_variance,
                                             transform=Compose([MedfilterChannel(width=21, height=7, channel_idx=0),
                                                                TripleChanne1(),
                                                                ToTensor()])),
                          batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=3)

    def lb(self, batch_size=128):
        return DataLoader(dataset=LB_Dataset(is_divide_variance=self.is_divide_variance,
                                             transform=Compose([MedfilterChannel(width=21, height=7, channel_idx=0),
                                                                TripleChanne1(),
                                                                ToTensor()])),
                          batch_size=batch_size, shuffle=False, num_workers=3)


class LB_MeanSub_Loader:
    def __init__(self, is_divide_variance=True):
        self.is_divide_variance = is_divide_variance

    def train(self, batch_size=128, shuffle=True):
        return DataLoader(dataset=TrainSet(is_divide_variance=self.is_divide_variance,
                                           transform=Compose([MeanSubtractionChannel(channel_idx=0),
                                                              TripleChanne1(),
                                                              ToTensor()])),
                          batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=3)

    def val(self, batch_size=128):
        return DataLoader(dataset=ValSet(is_divide_variance=self.is_divide_variance,
                                         transform=Compose([MeanSubtractionChannel(channel_idx=0),
                                                            TripleChanne1(),
                                                            ToTensor()])),
                          batch_size=batch_size, shuffle=False, num_workers=3)

    def dev(self, batch_size=128, shuffle=True):
        return DataLoader(dataset=DevDataSet(is_divide_variance=self.is_divide_variance,
                                             transform=Compose([MeanSubtractionChannel(channel_idx=0),
                                                                TripleChanne1(),
                                                                ToTensor()])),
                          batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=3)

    def lb(self, batch_size=128):
        return DataLoader(dataset=LB_Dataset(is_divide_variance=self.is_divide_variance,
                                             transform=Compose([MeanSubtractionChannel(channel_idx=0),
                                                                TripleChanne1(),
                                                                ToTensor()])),
                          batch_size=batch_size, shuffle=False, num_workers=3)


if __name__ == '__main__':
    loader = LB_Loader()
    dev = loader.train(batch_size=128)
    for i, x in enumerate(dev):
        print(x.size())