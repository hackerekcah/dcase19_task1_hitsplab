from torch.utils.data import DataLoader
from data_loader.transformer import *
from torchvision.transforms import Compose
from data_loader.data_sets import EVA_Dataset, TrainSet, ValSet, DeviceWiseValSet


class EVA_Loader:
    def __init__(self, is_divide_variance=True):
        self.is_divide_variance = is_divide_variance

    def train(self, batch_size=128):
        return DataLoader(dataset=TrainSet(is_divide_variance=self.is_divide_variance,
                                           transform=Compose([TripleChanne1(), ToTensor()])),
                          batch_size=batch_size, shuffle=False, drop_last=False, num_workers=3)

    def val(self, batch_size=128):
        return DataLoader(dataset=ValSet(is_divide_variance=self.is_divide_variance,
                                         transform=Compose([TripleChanne1(), ToTensor()])),
                          batch_size=batch_size, shuffle=False, num_workers=3)

    def eva(self, batch_size=128):
        return DataLoader(dataset=EVA_Dataset(is_divide_variance=self.is_divide_variance,
                                             transform=Compose([TripleChanne1(),
                                                                ToTensor()])),
                          batch_size=batch_size, shuffle=False, num_workers=3)


class EVA_Medfilter_Loader:
    def __init__(self, is_divide_variance=True):
        self.is_divide_variance = is_divide_variance

    def train(self, batch_size=128):
        return DataLoader(dataset=TrainSet(is_divide_variance=self.is_divide_variance,
                                           transform=Compose([MedfilterChannel(width=21, height=7, channel_idx=0),
                                                              TripleChanne1(), ToTensor()])),
                          batch_size=batch_size, shuffle=False, drop_last=False, num_workers=3)

    def val(self, batch_size=128):
        return DataLoader(dataset=ValSet(is_divide_variance=self.is_divide_variance,
                                         transform=Compose([MedfilterChannel(width=21, height=7, channel_idx=0),
                                                            TripleChanne1(), ToTensor()])),
                          batch_size=batch_size, shuffle=False, num_workers=3)

    def eva(self, batch_size=128):
        return DataLoader(dataset=EVA_Dataset(is_divide_variance=self.is_divide_variance,
                                             transform=Compose([MedfilterChannel(width=21, height=7, channel_idx=0),
                                                                TripleChanne1(),
                                                                ToTensor()])),
                          batch_size=batch_size, shuffle=False, num_workers=3)


class EVA_MeanSub_Loader:
    def __init__(self, is_divide_variance=True):
        self.is_divide_variance = is_divide_variance

    def train(self, batch_size=128):
        return DataLoader(dataset=TrainSet(is_divide_variance=self.is_divide_variance,
                                           transform=Compose([MeanSubtractionChannel(channel_idx=0),
                                                              TripleChanne1(), ToTensor()])),
                          batch_size=batch_size, shuffle=False, drop_last=False, num_workers=3)

    def val(self, batch_size=128):
        return DataLoader(dataset=ValSet(is_divide_variance=self.is_divide_variance,
                                         transform=Compose([MeanSubtractionChannel(channel_idx=0),
                                                            TripleChanne1(), ToTensor()])),
                          batch_size=batch_size, shuffle=False, num_workers=3)

    def eva(self, batch_size=128):
        return DataLoader(dataset=EVA_Dataset(is_divide_variance=self.is_divide_variance,
                                             transform=Compose([MeanSubtractionChannel(channel_idx=0),
                                                                TripleChanne1(),
                                                                ToTensor()])),
                          batch_size=batch_size, shuffle=False, num_workers=3)


class Device_Wise_Val_Loader:
    def __init__(self, is_divide_variance=True):
        self.is_divide_variance = is_divide_variance

    def val(self, batch_size=128):
        val_a = DataLoader(dataset=DeviceWiseValSet(is_divide_variance=self.is_divide_variance, device='a',
                                                    transform=Compose([TripleChanne1(), ToTensor()])),
                           batch_size=batch_size, shuffle=False, num_workers=3)
        val_b = DataLoader(dataset=DeviceWiseValSet(is_divide_variance=self.is_divide_variance, device='b',
                                                    transform=Compose([TripleChanne1(), ToTensor()])),
                           batch_size=batch_size, shuffle=False, num_workers=3)
        val_c = DataLoader(dataset=DeviceWiseValSet(is_divide_variance=self.is_divide_variance, device='c',
                                                    transform=Compose([TripleChanne1(), ToTensor()])),
                           batch_size=batch_size, shuffle=False, num_workers=3)
        return [val_a, val_b, val_c]


class Device_Wise_Medfilter_Val_Loader:
    def __init__(self, is_divide_variance=True):
        self.is_divide_variance = is_divide_variance

    def val(self, batch_size=128):
        val_a = DataLoader(dataset=DeviceWiseValSet(is_divide_variance=self.is_divide_variance, device='a',
                                                    transform=Compose([MedfilterChannel(width=21, height=7, channel_idx=0),
                                                                       TripleChanne1(), ToTensor()])),
                           batch_size=batch_size, shuffle=False, num_workers=3)
        val_b = DataLoader(dataset=DeviceWiseValSet(is_divide_variance=self.is_divide_variance, device='b',
                                                    transform=Compose([MedfilterChannel(width=21, height=7, channel_idx=0),
                                                                       TripleChanne1(), ToTensor()])),
                           batch_size=batch_size, shuffle=False, num_workers=3)
        val_c = DataLoader(dataset=DeviceWiseValSet(is_divide_variance=self.is_divide_variance, device='c',
                                                    transform=Compose([MedfilterChannel(width=21, height=7, channel_idx=0),
                                                                       TripleChanne1(), ToTensor()])),
                           batch_size=batch_size, shuffle=False, num_workers=3)
        return [val_a, val_b, val_c]

class Device_Wise_MeanSub_Val_Loader:
    def __init__(self, is_divide_variance=True):
        self.is_divide_variance = is_divide_variance

    def val(self, batch_size=128):
        val_a = DataLoader(dataset=DeviceWiseValSet(is_divide_variance=self.is_divide_variance, device='a',
                                                    transform=Compose([MeanSubtractionChannel(channel_idx=0),
                                                                       TripleChanne1(), ToTensor()])),
                           batch_size=batch_size, shuffle=False, num_workers=3)
        val_b = DataLoader(dataset=DeviceWiseValSet(is_divide_variance=self.is_divide_variance, device='b',
                                                    transform=Compose([MeanSubtractionChannel(channel_idx=0),
                                                                       TripleChanne1(), ToTensor()])),
                           batch_size=batch_size, shuffle=False, num_workers=3)
        val_c = DataLoader(dataset=DeviceWiseValSet(is_divide_variance=self.is_divide_variance, device='c',
                                                    transform=Compose([MeanSubtractionChannel(channel_idx=0),
                                                                       TripleChanne1(), ToTensor()])),
                           batch_size=batch_size, shuffle=False, num_workers=3)
        return [val_a, val_b, val_c]


if __name__ == '__main__':
    loader = EVA_Loader()
    dev = loader.eva(batch_size=128)
    for i, x in enumerate(dev):
        print(x.size())