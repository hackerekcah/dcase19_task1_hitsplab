import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import os


class History (object):
    """
    class to hold `loss` and `accuracy` over `epoch`
    """

    def __init__(self, name=None):
        self.name = name
        self.epoch = []
        self.loss = []
        self.acc = []
        self.axes = []
        self.recent = None

    def add(self, logs, epoch):
        self.recent = logs
        self.epoch.append(epoch)
        self.loss.append(logs['loss'])
        self.acc.append(logs['acc'])

    def set_axes(self, axes=None):
        if axes:
            self.axes = axes
        # new figure and axis
        else:
            self.axes = []
            plt.figure()
            self.axes.append(plt.subplot(2, 1, 1))
            self.axes.append(plt.subplot(2, 1, 2))

    def _get_tick(self):
        tick_max = np.max(self.epoch)
        ticks_int = np.arange(0, tick_max, np.ceil(tick_max / 5))
        if max(ticks_int) != tick_max:
            ticks_int = np.append(ticks_int, tick_max)
        return ticks_int

    def plot(self, axes=None, show=True):
        """
        plot loss acc in subplots
        :param axes: # axes usually returned by subplot if provided
        :param show:
        :return:
        """
        # if provided, set, else create
        self.set_axes(axes=axes)

        ticks = self._get_tick()
        if self.loss is not None:
            self.axes[0].plot(self.epoch, self.loss)
            self.axes[0].legend([self.name + "/loss"])
            self.axes[0].set_xticks(ticks)
            self.axes[0].set_xticklabels([str(e) for e in ticks])
        if self.acc is not None:
            self.axes[1].plot(self.epoch, self.acc)
            self.axes[1].legend([self.name + "/acc"])
            self.axes[1].set_xticks(ticks)
            self.axes[1].set_xticklabels([str(e) for e in ticks])

        plt.show() if show else None

    def clc_plot(self, axes=None, show=True):
        """
        clear output before plot, use in jupyter notebook to dynamically plot
        :param axes:
        :param show:
        :return:
        """
        clear_output(wait=True)
        self.plot(axes=axes, show=show)

    def clear(self):
        clear_output(wait=True)


class Reporter(object):
    def __init__(self, exp, ckpt_file=None):
        self.ckpt_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/ckpt'
        self.exp_path = os.path.join(self.ckpt_root, exp)
        self.run_list = os.listdir(self.exp_path)
        self.selected_ckpt = None
        self.selected_epoch = None
        self.selected_log = None
        self.selected_run = None

    def select_best(self, run=""):

        """
        set self.selected_run, self.selected_ckpt, self.selected_epoch
        :param run:
        :return:
        """

        matched = []
        for fname in self.run_list:
            if fname.startswith(run) and fname.endswith('tar'):
                matched.append(fname)

        acc = []
        import re
        for s in matched:
            acc_str = re.search('acc_(.*)\.tar', s).group(1)
            acc.append(float(acc_str))

        best_idx = np.argmax(acc)
        best_fname = matched[best_idx]

        self.selected_run = best_fname.split(',')[0]
        self.selected_epoch = int(re.search('Epoch_(.*),acc', best_fname).group(1))

        ckpt_file = os.path.join(self.exp_path, best_fname)

        self.selected_ckpt = ckpt_file

        return self

    def select_last(self, run):
        assert run and run != ""
        self.selected_run = run
        matched = []
        for fname in self.run_list:
            if fname.startswith(run) and fname.endswith('tar'):
                matched.append(fname)
        epochs = []
        import re
        for s in matched:
            epoch_str = re.search('Epoch_(.*),acc', s).group(1)
            epochs.append(int(epoch_str))
        last_idx = np.argmax(epochs)
        last_fname = matched[last_idx]

        self.selected_epoch = int(re.search('Epoch_(.*),acc', last_fname).group(1))

        ckpt_file = os.path.join(self.exp_path, last_fname)

        self.selected_ckpt = ckpt_file
        return self

    def pconfig(self, run):
        if run:
            self.selected_log = os.path.join(self.exp_path, run+'.log')
        else:
            self.selected_log = os.path.join(self.exp_path, self.selected_run + '.log')

        with open(self.selected_log, 'r') as f:
            config = f.readline()
            # config = config.split('\n')[0]
            print("<<<Config:\n{}".format(config))

    def _print(self, run, epoch):
        if run and epoch:
            print("<<<Select: Run:{}, Epoch:{}\n".format(run, epoch))

        self.pconfig(run=run)
        import torch
        ckpt = torch.load(self.selected_ckpt)
        histories = ckpt['histories']
        print("<<<Result")
        for hist in histories:
            if run and epoch:
                print("{:10},Epoch{:4d},Acc{:.3f}".format(hist.name, epoch,
                                                          hist.acc[epoch-1]))
            else:
                print("{:10},Epoch{:4d},Acc{:.3f}".format(hist.name, self.selected_epoch,
                                                          hist.acc[self.selected_epoch - 1]))

    def pbest(self, run="", epoch=None):
        self.select_best(run=run)
        print("<<<Best: Run:{}, Epoch:{}\n".format(self.selected_run, self.selected_epoch))
        self._print(run=run, epoch=epoch)

    def plast(self, run, epoch=None):
        self.select_last(run=run)
        print("<<<Last: Run:{}, Epoch:{}\n".format(self.selected_run, self.selected_epoch))
        self._print(run=run, epoch=epoch)

