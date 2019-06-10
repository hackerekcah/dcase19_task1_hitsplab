import torch
from data_loader.lb_loader import *
import argparse
import os
import numpy as np


class Submitter:
    def __init__(self):
        import csv
        csv_file = open('submittion.csv', 'a')
        writer = csv.writer(csv_file)
        writer.writerow(['Id', 'Scene_label'])
        self.writer = writer
        self.counter = 0

        self.labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
                       'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

    def submit_batch(self, pred_batch):
        for label_int in pred_batch:
            self.writer.writerow([str(self.counter), self.labels[label_int]])
            self.counter = self.counter + 1


def submit(pred_int):

    labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
              'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

    import csv
    with open('submittion.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Id', 'Scene_label'])
        for i in range(len(pred_int)):
            writer.writerow([str(i), labels[pred_int[i]]])


def lb_predict(args):

    # set up cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda')

    lb_loader = LB_Loader(args.is_divide_variance).lb(batch_size=args.batch_size)

    from xception import ModifiedXception
    model = ModifiedXception(num_classes=args.nb_class, drop_rate=args.drop_rate, decay=args.decay)

    model.load_state_dict(torch.load(args.ckpt_file)['model_state_dict'])

    model.to(device)

    model.eval()

    submitter = Submitter()

    for x in lb_loader:
        x = x.to(device)
        batch_pred = model(x)
        pred_int = torch.argmax(batch_pred, dim=1)
        submitter.submit_batch(pred_int.cpu().numpy())


def get_pred(model, loader, device):
    # model.eval()
    pred = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            batch_prob = model(x)
            # import torch.nn.functional as F
            # batch_prob = F.softmax(batch_prob, dim=1)
            batch_prob = batch_prob.cpu().detach().numpy()
            # move to cpu to release gpu prob
            pred.append(batch_prob)

    return np.concatenate(pred, 0)


def combine_predict(args):

    # set up cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda')

    lb_loader = LB_Loader(args.is_divide_variance).lb(batch_size=args.batch_size)

    from xception import ModifiedXception
    model = ModifiedXception(num_classes=args.nb_class, drop_rate=args.drop_rate, decay=args.decay)

    model.load_state_dict(torch.load(args.ckpt_file)['model_state_dict'])
    model.to(device)
    model.eval()
    prob1 = get_pred(model, lb_loader, device)

    model.load_state_dict(torch.load(args.ckpt_file1)['model_state_dict'])
    model.to(device).eval()

    prob2 = get_pred(model, lb_loader, device)

    prob = prob1 + prob2

    pred_int = np.argmax(prob, axis=1)

    submitter = Submitter()

    submitter.submit_batch(pred_int)


def combine_predict3(args):

    # set up cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda')

    lb_loader1 = LB_Loader(is_divide_variance=False).lb(batch_size=args.batch_size)
    lb_loader2 = LB_Medfilter_Loader(is_divide_variance=False).lb(batch_size=args.batch_size)
    lb_loader3 = LB_MeanSub_Loader(is_divide_variance=False).lb(batch_size=args.batch_size)

    from xception import ModifiedXception
    model = ModifiedXception(num_classes=args.nb_class, drop_rate=args.drop_rate, decay=args.decay)

    model.load_state_dict(torch.load(args.ckpt_file)['model_state_dict'], strict=False)
    model.to(device).eval()
    prob1 = get_pred(model, lb_loader1, device)

    model.load_state_dict(torch.load(args.ckpt_file1)['model_state_dict'])
    model.to(device).eval()

    prob2 = get_pred(model, lb_loader2, device)

    model.load_state_dict(torch.load(args.ckpt_file2)['model_state_dict'])
    model.to(device).eval()

    prob3 = get_pred(model, lb_loader3, device)

    prob = prob1 + prob2 + prob3

    pred_int = np.argmax(prob, axis=1)

    submitter = Submitter()

    submitter.submit_batch(pred_int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='3', type=str)
    parser.add_argument('--nb_class', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--decay', default=0.8, type=float)
    parser.add_argument('--drop_rate', default=0.3, type=float)
    parser.add_argument('--ckpt_file',
                        default='ckpt/xcep_mixup/Run01,ModifiedXception,Epoch_38,acc_0.753086.tar',
                        type=str)
    parser.add_argument('--ckpt_file1',
                        default='ckpt/medfilter_xcep_mixup/Run01,ModifiedXception,Epoch_40,acc_0.718708.tar',
                        type=str)
    parser.add_argument('--ckpt_file2',
                        default='ckpt/meansub_xcep_mixup/Run01,ModifiedXception,Epoch_38,acc_0.691358.tar',
                        type=str)
    parser.add_argument('--is_divide_variance', default=False, type=bool)
    args = parser.parse_args()
    # lb_predict(args)
    # combine_predict(args)
    combine_predict3(args)
