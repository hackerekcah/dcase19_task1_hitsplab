import argparse
import torch
import os
from data_loader.lb_loader import LB_Loader
import torch.optim as optim
from torch.optim.lr_scheduler import *
from engine import *
from utils.check_point import CheckPoint
from utils.history import History, Reporter
import numpy as np
import logging
from utils.utilities import set_seed, set_logging
torch.backends.cudnn.benchmark = True
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(args):

    set_seed(args.seed)

    set_logging(ROOT_DIR, args)
    logging.info(str(args))

    # set up cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda')

    loader = LB_Loader(is_divide_variance=args.is_divide_variance)

    train_loader = loader.train(batch_size=args.batch_size)
    val_loader = loader.val(batch_size=args.batch_size)

    # model = getattr(net_archs, args.net)(args).cuda()
    from xception import ModifiedXception
    model = ModifiedXception(num_classes=args.nb_class, drop_rate=args.drop_rate, decay=args.decay).cuda()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.l2)
    if args.lr_factor < 1.0:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True,
                                      factor=args.lr_factor, patience=args.lr_patience)

    train_hist, val_hist = History(name='train'), History(name='val')

    if args.continue_run:
        ckpt_file = Reporter(exp=args.exp).select_last(args.ckpt_prefix[0:5]).selected_ckpt
        logging.info('continue training from {}'.format(ckpt_file))

        ckpt_dicts = torch.load(ckpt_file)

        model.load_state_dict(ckpt_dicts['model_state_dict'])
        model.cuda()

        optimizer.load_state_dict(ckpt_dicts['optimizer_state_dict'])

        start_epoch = ckpt_dicts['epoch'] + 1
    else:
        start_epoch = 1

    # checkpoint after new History, order matters
    ckpter = CheckPoint(model=model, optimizer=optimizer, path='{}/ckpt/{}'.format(ROOT_DIR, args.exp),
                        prefix=args.ckpt_prefix, interval=1, save_num=1)

    for epoch in range(start_epoch, args.run_epochs):

        train_mixup_all(train_loader, model, optimizer, device, mix_alpha=args.mix_alpha)

        train_hist.add(
            logs=eval_model(train_loader, model, device),
            epoch=epoch
        )
        val_hist.add(
            logs=eval_model(val_loader, model, device),
            epoch=epoch
        )
        if args.lr_factor < 1.0:
            scheduler.step(val_hist.recent['acc'])

        # plotting
        if args.plot:
            train_hist.clc_plot()
            val_hist.plot()

        # logging
        logging.info("Epoch{:04d},{:6},{}".format(epoch, train_hist.name, str(train_hist.recent)))
        logging.info("Epoch{:04d},{:6},{}".format(epoch, val_hist.name, str(val_hist.recent)))

        ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=val_hist.recent)

    # explicitly save last
    ckpter.save(epoch=args.run_epochs-1, monitor='acc', loss_acc=val_hist.recent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='temp', type=str)
    parser.add_argument('--net', default='ModifiedXception', type=str)
    parser.add_argument('--ckpt_prefix', default='Run01', type=str)
    parser.add_argument('--device', default='5', type=str)
    parser.add_argument('--run_epochs', default=10, type=int)
    parser.add_argument('--nb_class', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--init_lr', default=3e-4, type=float)
    parser.add_argument('--lr_patience', default=3, type=int)
    parser.add_argument('--lr_factor', default=0.5, type=float)
    parser.add_argument('--plot', default=False, type=bool)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--decay', default=0.998, type=float)
    parser.add_argument('--drop_rate', default=0.3, type=float)
    parser.add_argument('--mix_alpha', default=0.1, type=float)
    parser.add_argument('--continue_run', default=False, type=bool)
    parser.add_argument('--is_divide_variance', default=True, type=bool)
    args = parser.parse_args()
    run(args)