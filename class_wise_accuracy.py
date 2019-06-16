import torch
import logging
from data_loader.eva_loader import *
import argparse
import os
import numpy as np
import sys
import torch.nn.functional as F
from utils.utilities import calculate_accuracy, plot_confusion_matrix, calculate_confusion_matrix
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_pred_prob(model, loader, device, using_softmax=False):
    model.eval()
    pred = []
    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(device)
            batch_prob = model(x)
            if using_softmax:
                batch_prob = F.softmax(batch_prob, dim=1)
            batch_prob = batch_prob.cpu().detach().numpy()
            pred.append(batch_prob)

    return np.concatenate(pred, 0)


def get_target(loader):
    target = []
    for (x, y) in loader:
        _, y = y.max(dim=1)
        target.append(y)
    return np.concatenate(target)


def get_class_wise_accuracy(args):
    # single model

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda')

    if args.method == 'normal':
        loader = EVA_Loader(is_divide_variance=args.model1_is_divide_variance)
    elif args.method == 'medfilter':
        loader = EVA_Medfilter_Loader(is_divide_variance=args.model2_is_divide_variance)
    elif args.method == 'meansub':
        loader = EVA_MeanSub_Loader(is_divide_variance=args.model3_is_divide_variance)
    else:
        print('Error. Please choose one of ["normal", "medfilter", "meansub"]')
        return None

    train_loader = loader.train(batch_size=args.batch_size)
    val_loader = loader.val(batch_size=args.batch_size)

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
    train_acc = (np.array(train_target) == np.array(train_pred)).mean()

    val_acc_class_wise = calculate_accuracy(target=val_target, predict=val_pred, classes_num=args.nb_class)
    val_acc = (np.array(val_target) == np.array(val_pred)).mean()

    labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
              'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

    train_acc_cw = dict()
    val_acc_cw = dict()
    for i in range(args.nb_class):
        train_acc_cw[labels[i]] = train_acc_class_wise[i]
        val_acc_cw[labels[i]] = val_acc_class_wise[i]

    return train_acc_cw, val_acc_cw, train_acc, val_acc


def get_loaders(batch_size, model1_var=True, model2_var=False, model3_var=False):
    loaders = []
    loaders.append(Device_Wise_Val_Loader(is_divide_variance=model1_var).val(batch_size=batch_size))
    loaders.append(Device_Wise_Medfilter_Val_Loader(is_divide_variance=model2_var).val(batch_size=batch_size))
    loaders.append(Device_Wise_MeanSub_Val_Loader(is_divide_variance=model3_var).val(batch_size=batch_size))

    return loaders


def get_device_wise_accuracy_multi_models(args):
    # device-wise, class-wise, multi-model fusion

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda')

    loaders = get_loaders(batch_size=args.batch_size, model1_var=args.model1_is_divide_variance,
                          model2_var=args.model2_is_divide_variance, model3_var=args.model3_is_divide_variance)

    from xception import ModifiedXception

    labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
              'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']
    val_acc_cw = []
    val_acc_mean = []
    bc_val_pred = []
    bc_val_target = []
    for i in range(3):
        model = ModifiedXception(num_classes=args.nb_class, drop_rate=args.drop_rate, decay=args.decay)
        model.load_state_dict(torch.load(os.path.join(ROOT_DIR, args.ckpt_file))['model_state_dict'])
        model.to(device)
        val_pred_prob1 = get_pred_prob(model, loaders[0][i], device)
        model = ModifiedXception(num_classes=args.nb_class, drop_rate=args.drop_rate, decay=args.decay)
        model.load_state_dict(torch.load(os.path.join(ROOT_DIR, args.ckpt_file1))['model_state_dict'])
        model.to(device)
        val_pred_prob2 = get_pred_prob(model, loaders[1][i], device)
        model = ModifiedXception(num_classes=args.nb_class, drop_rate=args.drop_rate, decay=args.decay)
        model.load_state_dict(torch.load(os.path.join(ROOT_DIR, args.ckpt_file2))['model_state_dict'])
        model.to(device)
        val_pred_prob3 = get_pred_prob(model, loaders[2][i], device)
        val_pred_prob = val_pred_prob1 + val_pred_prob2 + val_pred_prob3
        val_pred = np.argmax(val_pred_prob, axis=1)

        val_target = get_target(loaders[0][i])

        val_acc = calculate_accuracy(target=val_target, predict=val_pred, classes_num=args.nb_class)
        if i > 0:
            bc_val_pred.append(val_pred)
            bc_val_target.append(val_target)
        acc_mean = (np.array(val_target) == np.array(val_pred)).mean()
        val_acc_mean.append(acc_mean)

        val_acc_class_wise = dict()
        for i in range(args.nb_class):
            val_acc_class_wise[labels[i]] = val_acc[i]
        val_acc_cw.append(val_acc_class_wise)

    # average of device (b, c)
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


def get_device_wise_accuracy(args):
    # single model

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda')

    if args.method == 'normal':
        loader = Device_Wise_Val_Loader(is_divide_variance=args.model1_is_divide_variance)
    elif args.method == 'medfilter':
        loader = Device_Wise_Medfilter_Val_Loader(is_divide_variance=args.model2_is_divide_variance)
    elif args.method == 'meansub':
        loader = Device_Wise_MeanSub_Val_Loader(is_divide_variance=args.model3_is_divide_variance)
    else:
        print('Error. Please choose one of ["normal", "medfilter", "meansub"]')
        return None

    val_loader = loader.val(batch_size=args.batch_size)

    from xception import ModifiedXception
    model = ModifiedXception(num_classes=args.nb_class, drop_rate=args.drop_rate, decay=args.decay)
    model.load_state_dict(torch.load(os.path.join(ROOT_DIR, args.ckpt_file))['model_state_dict'])
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


def print_accuracy(args):

    set_acc_logging(args)
    import pprint
    logging.info(pprint.pformat(vars(args)) if not isinstance(args, dict) else pprint.pformat(args))

    if args.model == 'single':
        val_acc_cw, val_acc = get_device_wise_accuracy(args)
    elif args.model == 'multi':
        val_acc_cw, val_acc = get_device_wise_accuracy_multi_models(args)
    else:
        print('model type input error!')
        return
    device_list = ['device_A', 'device_B', 'device_C', 'Average (B, C)']
    for i in range(4):
        logging.info('{}:'.format(device_list[i]))
        logging.info('{} Validation set average accuracy: {:.3f}'.format(device_list[i], val_acc[i]))
        logging.info('{} Validation set class-wise accuracy:'.format(device_list[i]))
        for key in val_acc_cw[i].keys():
            logging.info('{:22s}: {:.3f}'.format(key, val_acc_cw[i][key]))


def set_acc_logging(args):

    if args.model == 'single':
        # acc log file name RunXX.acc, same path as ckpt file
        log_file = os.path.join(ROOT_DIR, args.ckpt_file.split(',')[0] + '.acc')
    else:
        log_file = os.path.join(ROOT_DIR, args.acc_log_file)

    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    elif os.path.exists(log_file):
        print('acc_log_file:{} already exists, program stop!')
        sys.exit()

    formatter = logging.Formatter('%(message)s')

    # output to file
    fileh = logging.FileHandler(log_file, 'a')
    fileh.setLevel('INFO')
    fileh.setFormatter(formatter)

    # output to stdout
    streamh = logging.StreamHandler(sys.stdout)
    streamh.setLevel('INFO')
    streamh.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    log.setLevel('INFO')
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)

    # set new handler
    log.addHandler(fileh)
    log.addHandler(streamh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str)
    parser.add_argument('--nb_class', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--decay', default=1.0, type=float)
    parser.add_argument('--drop_rate', default=0.3, type=float)
    parser.add_argument('--method', default='normal', type=str)
    parser.add_argument('--model', default='single', type=str)
    parser.add_argument('--ckpt_file2',
                        default='ckpt/meansub_xcep_mixup/Run01,ModifiedXception,Epoch_38,acc_0.691358.tar',
                        type=str)
    parser.add_argument('--ckpt_file1',
                        default='ckpt/medfilter_xcep_mixup/Run01,ModifiedXception,Epoch_40,acc_0.718708.tar',
                        type=str)
    parser.add_argument('--ckpt_file',
                        default='ckpt/xcep_mixup/Run01,ModifiedXception,Epoch_68,acc_0.748528.tar',
                        type=str)
    parser.add_argument('--model1_is_divide_variance', default=True, type=bool)
    parser.add_argument('--model2_is_divide_variance', default=False, type=bool)
    parser.add_argument('--model3_is_divide_variance', default=False, type=bool)

    parser.add_argument('--acc_log_file', default='submissions/acc/run.acc', type=str)
    args = parser.parse_args()

    print_accuracy(args)
