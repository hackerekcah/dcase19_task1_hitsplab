import torch
from data_loader.eva_loader import *
import argparse
import os
import numpy as np
from utils.utilities import calculate_accuracy, plot_confusion_matrix, calculate_confusion_matrix


def get_pred_prob(model, loader, device, using_softmax=False):
    model.eval()
    pred = []
    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(device)
            batch_prob = model(x)
            if using_softmax:
                import torch.nn.functional as F
                batch_prob = F.softmax(batch_prob, dim=1)
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
    # TODO more accurate
    train_acc = (np.array(train_target) == np.array(train_pred)).mean()
    # train_acc = np.mean(train_acc_class_wise)

    val_acc_class_wise = calculate_accuracy(target=val_target, predict=val_pred, classes_num=args.nb_class)
    val_acc = (np.array(val_target) == np.array(val_pred)).mean()
    # val_acc = np.mean(val_acc_class_wise)

    labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
              'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

    train_acc_cw = dict()
    val_acc_cw = dict()
    for i in range(args.nb_class):
        train_acc_cw[labels[i]] = train_acc_class_wise[i]
        val_acc_cw[labels[i]] = val_acc_class_wise[i]

    return train_acc_cw, val_acc_cw, train_acc, val_acc


def get_loaders(batch_size):
    loaders = []
    loaders.append(Device_Wise_Val_Loader(is_divide_variance=True).val(batch_size=batch_size))
    loaders.append(Device_Wise_Medfilter_Val_Loader(is_divide_variance=False).val(batch_size=batch_size))
    loaders.append(Device_Wise_MeanSub_Val_Loader(is_divide_variance=False).val(batch_size=batch_size))

    return loaders


def get_device_accuracy_multi_models(args):

    # set up cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda')

    loaders = get_loaders(batch_size=args.batch_size)

    from xception import ModifiedXception

    labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
              'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']
    val_acc_cw = []
    val_acc_mean = []
    val_acc_bc = []
    for i in range(3):
        model = ModifiedXception(num_classes=args.nb_class, drop_rate=args.drop_rate, decay=args.decay)
        model.load_state_dict(torch.load(args.ckpt_file)['model_state_dict'])
        model.to(device)
        val_pred_prob1 = get_pred_prob(model, loaders[0][i], device)
        model = ModifiedXception(num_classes=args.nb_class, drop_rate=args.drop_rate, decay=args.decay)
        model.load_state_dict(torch.load(args.ckpt_file1)['model_state_dict'])
        model.to(device)
        val_pred_prob2 = get_pred_prob(model, loaders[1][i], device)
        model = ModifiedXception(num_classes=args.nb_class, drop_rate=args.drop_rate, decay=args.decay)
        model.load_state_dict(torch.load(args.ckpt_file2)['model_state_dict'])
        model.to(device)
        val_pred_prob3 = get_pred_prob(model, loaders[2][i], device)
        val_pred_prob = val_pred_prob1 + val_pred_prob2 + val_pred_prob3
        val_pred = np.argmax(val_pred_prob, axis=1)
        val_target = get_target(loaders[0][i])

        val_acc = calculate_accuracy(target=val_target, predict=val_pred, classes_num=args.nb_class)
        if i > 0:
            val_acc_bc.append(val_acc)
        acc_mean = (np.array(val_target) == np.array(val_pred)).mean()
        val_acc_mean.append(acc_mean)

        val_acc_class_wise = dict()
        for i in range(args.nb_class):
            val_acc_class_wise[labels[i]] = val_acc[i]
        val_acc_cw.append(val_acc_class_wise)

    bc_val_acc = (val_acc_bc[1] + val_acc_bc[0]) / 2
    val_acc_mean.append(np.mean(bc_val_acc))
    val_acc_class_wise = dict()
    for i in range(args.nb_class):
        val_acc_class_wise[labels[i]] = bc_val_acc[i]
    val_acc_cw.append(val_acc_class_wise)

    return val_acc_cw, val_acc_mean



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
    bc_val_pred = []
    bc_val_target = []
    for i in range(3):
        val_pred_prob = get_pred_prob(model, val_loader[i], device)
        val_pred = np.argmax(val_pred_prob, axis=1)
        val_target = get_target(val_loader[i])
        if i > 0:
            bc_val_pred.append(val_pred)
            bc_val_target.append(val_target)
        val_acc = calculate_accuracy(target=val_target, predict=val_pred, classes_num=args.nb_class)
        acc_mean = (np.array(val_target) == np.array(val_pred)).mean()
        val_acc_mean.append(acc_mean)

        val_acc_class_wise = dict()
        for i in range(args.nb_class):
            val_acc_class_wise[labels[i]] = val_acc[i]
        val_acc_cw.append(val_acc_class_wise)

    bc_val_target = np.concatenate(bc_val_target)
    bc_val_pred = np.concatenate(bc_val_pred)
    bc_val_acc = calculate_accuracy(target=bc_val_target, predict=bc_val_pred, classes_num=args.nb_class)
    bc_val_acc_mean = (np.array(bc_val_target) == np.array(bc_val_pred)).mean()
    val_acc_class_wise = dict()
    for i in range(args.nb_class):
        val_acc_class_wise[labels[i]] = bc_val_acc[i]
    val_acc_cw.append(val_acc_class_wise)
    val_acc_mean.append(bc_val_acc_mean)
    return val_acc_cw, val_acc_mean




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='3', type=str)
    parser.add_argument('--nb_class', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--decay', default=1.0, type=float)
    parser.add_argument('--drop_rate', default=0.3, type=float)
    parser.add_argument('--method', default='normal', type=str)
    parser.add_argument('--ckpt_file2',
                        default='ckpt/meansub_xcep_mixup/Run01,ModifiedXception,Epoch_38,acc_0.691358.tar',
                        type=str)
    parser.add_argument('--ckpt_file1',
                        default='ckpt/medfilter_xcep_mixup/Run01,ModifiedXception,Epoch_40,acc_0.718708.tar',
                        type=str)
    parser.add_argument('--ckpt_file',
                        default='ckpt/xcep_mixup/Run01,ModifiedXception,Epoch_38,acc_0.753086.tar',
                        type=str)
    parser.add_argument('--is_divide_variance', default=False, type=bool)
    args = parser.parse_args()

    # test of get_accuracy()
    # train_acc_cw, val_acc_cw, train_acc, val_acc = get_accuracy(args)
    # print('=' * 40)
    # print('Training set average accuracy: ', train_acc)
    # print('Training set class-wise accuray:')
    # for key in train_acc_cw.keys():
    #     print('{:22s}: {:.3f}'.format(key, train_acc_cw[key]))
    # print('Validation set average accuracy: ', val_acc)
    # for key in val_acc_cw.keys():
    #     print('{:22s}: {:.3f}'.format(key, val_acc_cw[key]))
    # print('=' * 40)

    # test of get_device_accuracy()
    # val_acc_cw, val_acc = get_device_accuracy(args)
    # device_list = ['device_A', 'device_B', 'device_C', 'Average (B, C)']
    # for i in range(4):
    #     print('{}:'.format(device_list[i]))
    #     print('{} Validation set average accuracy: {:.3f}'.format(device_list[i], val_acc[i]))
    #     print('{} Validation set class-wise accuracy:'.format(device_list[i]))
    #     for key in val_acc_cw[i].keys():
    #         print('{:22s}: {:.3f}'.format(key, val_acc_cw[i][key]))

    # eva_predict(args)
    # combine_predict(args)
    # combine_predict3(args)

    # ===================================================

    val_acc_cw, val_acc = get_device_accuracy_multi_models(args)
    # val_acc_cw, val_acc = get_device_accuracy(args)
    device_list = ['device_A', 'device_B', 'device_C', 'Average (B, C)']
    for i in range(4):
        print('{}:'.format(device_list[i]))
        print('{} Validation set average accuracy: {:.3f}'.format(device_list[i], val_acc[i]))
        print('{} Validation set class-wise accuracy:'.format(device_list[i]))
        for key in val_acc_cw[i].keys():
            print('{:22s}: {:.3f}'.format(key, val_acc_cw[i][key]))
