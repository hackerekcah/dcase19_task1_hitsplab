import torch
from data_loader.eva_loader import *
import argparse
import os
import numpy as np
from utils.utilities import calculate_accuracy, plot_confusion_matrix, calculate_confusion_matrix


def get_pred_prob(model, loader, device):
    model.eval()
    pred = []
    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(device)
            batch_prob = model(x)

            batch_prob = batch_prob.cpu().detach().numpy()
            # move to cpu to release gpu prob
            pred.append(batch_prob)

    return np.concatenate(pred, 0)


def get_target(loader):
    target = []
    for (x, y) in loader:
        _, y = y.max(dim=1)
        target.append(y)
    return np.concatenate(target)


def get_accuracy(args):

    # set up cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda')

    if args.method == 'normal':
        loader = EVA_Loader(is_divide_variance=args.is_divide_variance)
    elif args.method == 'medfilter':
        loader = EVA_Medfilter_Loader(is_divide_variance=args.is_divide_variance)
    elif args.method == 'meansub':
        loader = EVA_MeanSub_Loader(is_divide_variance=args.is_divicde_variance)
    else:
        print('Error. Please choose one of ["normal", "medfilter", "meansub"]')
        return None

    train_loader = loader.train(batch_size=64)
    val_loader = loader.val(batch_size=64)

    from xception import ModifiedXception
    model = ModifiedXception(num_classes=args.nb_class, drop_rate=args.drop_rate, decay=args.decay)
    model.load_state_dict(torch.load(args.ckpt_file)['model_state_dict'])
    model.to(device)

    train_pred_prob = get_pred_prob(model, train_loader, device)
    train_pred = np.argmax(train_pred_prob, axis=1)
    val_pred_prob = get_pred_prob(model, val_loader, device)
    val_pred = np.argmax(val_pred_prob, axis=1)

    train_target = get_target(train_loader)
    val_target = get_target(val_loader)

    train_acc_class_wise = calculate_accuracy(target=train_target, predict=train_pred, classes_num=args.nb_class)
    train_acc = np.mean(train_acc_class_wise)

    val_acc_class_wise = calculate_accuracy(target=val_target, predict=val_pred, classes_num=args.nb_class)
    val_acc = np.mean(val_acc_class_wise)

    labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
              'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

    train_acc_cw = dict()
    val_acc_cw = dict()
    for i in range(args.nb_class):
        train_acc_cw[labels[i]] = train_acc_class_wise[i]
        val_acc_cw[labels[i]] = val_acc_class_wise[i]

    return train_acc_cw, val_acc_cw, train_acc, val_acc


def get_device_accuracy(args):

    # set up cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda')

    if args.method == 'normal':
        loader = Device_Wise_Val_Loader(is_divide_variance=args.is_divide_variance)
    elif args.method == 'medfilter':
        loader = Device_Wise_Medfilter_Val_Loader(is_divide_variance=args.is_divide_variance)
    elif args.method == 'meansub':
        loader = Device_Wise_MeanSub_Val_Loader(is_divide_variance=args.is_divicde_variance)
    else:
        print('Error. Please choose one of ["normal", "medfilter", "meansub"]')
        return None

    val_loader = loader.val(batch_size=64)

    from xception import ModifiedXception
    model = ModifiedXception(num_classes=args.nb_class, drop_rate=args.drop_rate, decay=args.decay)
    model.load_state_dict(torch.load(args.ckpt_file)['model_state_dict'])
    model.to(device)

    labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
              'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']
    val_acc_cw = []
    val_acc_mean = []
    for i in range(3):
        val_pred_prob = get_pred_prob(model, val_loader[i], device)
        val_pred = np.argmax(val_pred_prob, axis=1)
        val_target = get_target(val_loader[i])
        val_acc = calculate_accuracy(target=val_target, predict=val_pred, classes_num=args.nb_class)
        val_acc_mean.append(np.mean(val_acc))
        val_acc_class_wise = dict()
        for i in range(args.nb_class):
            val_acc_class_wise[labels[i]] = val_acc[i]
        val_acc_cw.append(val_acc_class_wise)

    return val_acc_cw, val_acc_mean




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='2', type=str)
    parser.add_argument('--nb_class', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--decay', default=0.8, type=float)
    parser.add_argument('--drop_rate', default=0.3, type=float)
    parser.add_argument('--method', default='normal', type=str)
    parser.add_argument('--ckpt_file2',
                        default='ckpt/meansub_xcep_mixup/Run01,ModifiedXception,Epoch_35,acc_0.688319.tar',
                        type=str)
    parser.add_argument('--ckpt_file1',
                        default='ckpt/medfilter_xcep_mixup/Run01,ModifiedXception,Epoch_20,acc_0.712061.tar',
                        type=str)
    parser.add_argument('--ckpt_file',
                        default='ckpt/xcep_mixup/Run01,ModifiedXception,Epoch_38,acc_0.753086.tar',
                        type=str)
    parser.add_argument('--is_divide_variance', default=False, type=bool)
    args = parser.parse_args()

    # test of get_accuracy()
    train_acc_cw, val_acc_cw, train_acc, val_acc = get_accuracy(args)
    print('=' * 40)
    print('Training set average accuracy: ', train_acc)
    print('Training set class-wise accuray:')
    for key in train_acc_cw.keys():
        print('{:22s}: {:.3f}'.format(key, train_acc_cw[key]))
    print('Validation set average accuracy: ', val_acc)
    for key in val_acc_cw.keys():
        print('{:22s}: {:.3f}'.format(key, val_acc_cw[key]))
    print('=' * 40)

    # test of get_device_accuracy()
    val_acc_cw, val_acc = get_device_accuracy(args)
    device_list = ['device_A', 'device_B', 'device_C']
    for i in range(3):
        print('{}:'.format(device_list[i]))
        print('{} Validation set average accuracy: {:.3f}'.format(device_list[i], val_acc[i]))
        print('{} Validation set class-wise accuracy:'.format(device_list[i]))
        for key in val_acc_cw[i].keys():
            print('{:22s}: {:.3f}'.format(key, val_acc_cw[i][key]))

    # eva_predict(args)
    # combine_predict(args)
    # combine_predict3(args)
