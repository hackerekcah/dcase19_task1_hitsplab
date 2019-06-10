from torch.utils.data import Dataset
from data_loader.transformer import *
from data_manager.dcase19_lb_manager import Dcase19LBManager
from data_manager.dcase19_eva_manager import Dcase19EVAManager
from data_manager.dcase19_taskb import Dcase19TaskbData
from data_manager.dcase19_standrizer import DCASE19Standarizer


def load_dcase19_dev():
    from data_manager.dcase19_taskb import Dcase19TaskbData
    dev_manager = Dcase19TaskbData()
    train_x, train_y = dev_manager.load_dev(mode='train', devices='abc')
    test_x, test_y = dev_manager.load_dev(mode='test', devices='abc')
    x = np.concatenate([train_x, test_x], axis=0)
    y = np.concatenate([train_y, test_y], axis=0)
    return x, y


class TrainSet(Dataset):
    def __init__(self, is_divide_variance=True, transform=None):
        super(TrainSet, self).__init__()
        self.dev_manager = Dcase19TaskbData()
        dev_standrizer = DCASE19Standarizer(data_manager=self.dev_manager)
        if is_divide_variance:
            self.x, self.y = dev_standrizer.load_dev_standrized(mode='train', device='abc', norm_device='abc')
        else:
            self.x, self.y = dev_standrizer.load_dev_mean_subtracted(mode='train', device='abc', norm_device='abc')
        self.x = np.expand_dims(self.x, axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class ValSet(Dataset):
    def __init__(self, is_divide_variance=True, transform=None):
        super(ValSet, self).__init__()
        self.dev_manager = Dcase19TaskbData()
        dev_standrizer = DCASE19Standarizer(data_manager=self.dev_manager)
        if is_divide_variance:
            self.x, self.y = dev_standrizer.load_dev_standrized(mode='test', device='abc', norm_device='abc')
        else:
            self.x, self.y = dev_standrizer.load_dev_mean_subtracted(mode='test', device='abc', norm_device='abc')
        self.x = np.expand_dims(self.x, axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class LB_Dataset(Dataset):
    def __init__(self, is_divide_variance=True, transform=None):
        super(LB_Dataset, self).__init__()
        self.data_manager = Dcase19LBManager()
        # load data, normed by all device data
        self.x = self.data_manager.load_data()

        dev_x, _ = load_dcase19_dev()

        if is_divide_variance:
            # normalize lb data using dev
            from utils.utilities import standarize_with_norm
            self.x = standarize_with_norm([self.x], dev_x)[0]
        else:
            from utils.utilities import mean_sub_norm
            self.x = mean_sub_norm([self.x], dev_x)[0]

        self.x = np.expand_dims(self.x, axis=1)

        # self.x = np.repeat(self.x, repeats=3, axis=1)

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.transform(self.x[idx])


class DevDataSet(Dataset):
    def __init__(self, is_divide_variance=True, transform=None):
        super(DevDataSet, self).__init__()

        self.x, self.y = load_dcase19_dev()

        if is_divide_variance:
            # normalize x
            from utils.utilities import standarize_with_norm
            self.x = standarize_with_norm([self.x], self.x)[0]
        else:
            from utils.utilities import mean_sub_norm
            self.x = mean_sub_norm([self.x], self.x)[0]

        self.x = np.expand_dims(self.x, axis=1)

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class EVA_Dataset(Dataset):
    def __init__(self, is_divide_variance=True, transform=None):
        super(EVA_Dataset, self).__init__()
        self.data_manager = Dcase19EVAManager()
        # load data, normed by all device data
        self.x = self.data_manager.load_data()

        dev_x, _ = load_dcase19_dev()

        if is_divide_variance:
            # normalize lb data using dev
            from utils.utilities import standarize_with_norm
            self.x = standarize_with_norm([self.x], dev_x)[0]
        else:
            from utils.utilities import mean_sub_norm
            self.x = mean_sub_norm([self.x], dev_x)[0]

        self.x = np.expand_dims(self.x, axis=1)

        # self.x = np.repeat(self.x, repeats=3, axis=1)

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.transform(self.x[idx])


class DeviceWiseValSet(Dataset):
    def __init__(self, is_divide_variance=True, device='a', transform=None):
        super(DeviceWiseValSet, self).__init__()
        self.dev_manager = Dcase19TaskbData()
        dev_standrizer = DCASE19Standarizer(data_manager=self.dev_manager)
        if is_divide_variance:
            self.x, self.y = dev_standrizer.load_dev_standrized(mode='test', device=device, norm_device='abc')
        else:
            self.x, self.y = dev_standrizer.load_dev_mean_subtracted(mode='test', device=device, norm_device='abc')
        self.x = np.expand_dims(self.x, axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample
