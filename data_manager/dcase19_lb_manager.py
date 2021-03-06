import librosa
import h5py
import os
import numpy as np
import sys
from tqdm import tqdm
# need to explicit import display
import librosa.display
import matplotlib.pyplot as plt
from sklearn import preprocessing
import configparser

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


class Dcase19LBManager:

    def __init__(self):
        self.verbose = True
        config = configparser.ConfigParser()
        config.read(os.path.join(ROOT_DIR, 'data_manager.cfg'))
        self.config = config
        self.lb_path = config['dcase19_taskb']['lb_path']
        data_h5 = os.path.join(ROOT_DIR, 'data19_h5')
        if not os.path.exists(data_h5):
            os.makedirs(data_h5)

        self.lb_h5_path = os.path.join(data_h5, 'TaskbLB.h5')
        self.lb_matrix_h5_path = os.path.join(data_h5, 'TaskbLBMatrix.h5')

        self.meta_path = os.path.join(self.lb_path, 'evaluation_setup/test.csv')

    def extract_logmel(self, wav_path):
        """        self.fname_encoder = preprocessing.LabelEncoder()
        Give a wav, extract logmel feature
        :param wav_path:
        :return: fea of dim (1, frequency, time), first dim is added
        """
        x, sr = librosa.load(wav_path, sr=None, mono=False)
        assert (sr == 44100)

        # 40ms winlen, half overlap
        y = librosa.feature.melspectrogram(x,
                                           sr=int(self.config['logmel']['sr']),
                                           n_fft=int(self.config['logmel']['n_fft']),
                                           hop_length=int(self.config['logmel']['hop_length']),
                                           n_mels=int(self.config['logmel']['n_mels']),
                                           fmax=int(self.config['logmel']['fmax'])
                                           )
        # about 1e-7
        EPS = np.finfo(np.float32).eps
        fea = np.log(y+EPS)
        # add a new axis
        return np.expand_dims(fea[:, :-1], axis=0)

    def create_h5(self):
        """
        Extract LogMel and Store in h5 File, index by wav name
        :return:
        """
        if os.path.exists(self.lb_h5_path):
            print("[LOGGING]: " + self.lb_h5_path + " exists!")
            return

        with h5py.File(self.lb_h5_path, 'w') as f:

            # create a group: f['train']
            lb = f.create_group('lb')
            self.extract_fea_for_datagroup(lb)

        f.close()

    def extract_fea_for_datagroup(self, data_group):

        fp = open(self.meta_path, 'r')

        for i, line in tqdm(enumerate(fp)):
            if i == 0:
                # skip head line, which is : filename
                continue
            audio_name = line.split()[0]
            audio_path = os.path.join(self.lb_path, audio_name)
            fea = self.extract_logmel(wav_path=audio_path)

            wav_name = os.path.basename(audio_path)
            data_group[wav_name] = fea

    def create_lb_matrix(self):
        if os.path.exists(self.lb_matrix_h5_path):
            print("[LOGGING]: " + self.lb_matrix_h5_path + " exists!")
            return

        with h5py.File(self.lb_matrix_h5_path, 'w') as f:

            grp = f.create_group('lb')
            grp['data'] = self.extract_npy()

        f.close()

    def extract_npy(self):
        data = []
        with h5py.File(self.lb_h5_path, 'r') as f:
            audios = f['lb'].keys()
            # for audio in audios:
            #     data.append(np.array(f['lb'][audio].value))

            for i in range(len(audios)):
                wav_name = str(i) + '.wav'
                data.append(np.array(f['lb'][wav_name].value))
        # concat data along existing axis 0
        data = np.concatenate(data, axis=0)

        return data

    def load_data(self):

        if not os.path.exists(self.lb_matrix_h5_path):
            print(self.lb_matrix_h5_path + "not exists!")
            sys.exit()

        with h5py.File(self.lb_matrix_h5_path, 'r') as f:
            data = f['lb']['data'].value
            print("[LOGGING]: Loading leaderboard data of shape: ", data.shape)
            return data


if __name__ == '__main__':
    manager = Dcase19LBManager()
    manager.create_h5()
    manager.create_lb_matrix()
    manager.load_data()