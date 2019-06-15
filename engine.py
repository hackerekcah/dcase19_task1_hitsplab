import torch
from data_loader.mixup import mixup_data, mixup_criterion
import numpy as np
import logging


def train_uda_mixup(src_loader, dst_loader, model, optimizer, device, mix_alpha=0.1):
    """
    only reformulated form, mixup only data, not labels, fro semi-supervised
    :param src_loader:
    :param dst_loader:
    :param model:
    :param optimizer:
    :param device:
    :param mix_alpha:
    :return:
    """
    model.train(mode=True)
    for idx, ((x_src, y_src), (x_dst, _)) in enumerate(zip(src_loader, dst_loader)):
        x_src, y_src = x_src.to(device), y_src.to(device)
        x_dst = x_dst.to(device)

        # onehot to int
        _, y_src = y_src.max(dim=1)

        # asymmetric beta
        lam = np.random.beta(mix_alpha + 1, mix_alpha, len(x_src))
        lam = torch.from_numpy(lam)
        # .to() will copy data
        lam = lam.to(device=device, dtype=torch.float)
        # share the same data
        lamx = lam.view(-1, 1, 1, 1)
        mix_x = x_src * lamx + x_dst * (1. - lamx)
        logits = model(mix_x)

        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        loss = criterion(logits, y_src)

        loss_mean = torch.mean(loss)

        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()

def train_mixup_all(train_loader, model, optimizer, device, mix_alpha=0.1):
    model.train(mode=True)
    for idx, ((x1, y1), (x2, y2)) in enumerate(zip(train_loader, train_loader)):
        x1, y1 = x1.to(device), y1.to(device)
        x2, y2 = x2.to(device), y2.to(device)

        _, y1 = y1.max(dim=1)
        _, y2 = y2.max(dim=1)

        if mix_alpha <= 0.0:
            lam = np.zeros(len(x1))
        else:
            lam = np.random.beta(mix_alpha, mix_alpha, len(x1))

        lam = torch.from_numpy(lam)
        lam = lam.to(device, dtype=torch.float)
        lamx = lam.view(-1, 1, 1, 1)

        mix_x = x1 * lamx + x2 * (1. - lamx)
        logits = model(mix_x)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss = mixup_criterion(criterion, logits, y1, y2, lam)
        loss_mean = torch.mean(loss)

        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()


def sum_zero_loss(model, sz_penalty):
    # (out_channel, in_channel, height, width)
    conv1_weight = list(model.parameters())[0]
    conv1_weight = conv1_weight.view(conv1_weight.size(0), -1)
    sum_vec = torch.sum(conv1_weight, 1, keepdim=False)
    # loss = sz_penalty * torch.sum(sum_vec * sum_vec)
    loss = sz_penalty * torch.sum(torch.abs(sum_vec))
    return loss


def train_mixup_all_sumzero(train_loader, model, optimizer, device, mix_alpha=0.1, sz_penalty=1.0):
    model.train(mode=True)
    for idx, ((x1, y1), (x2, y2)) in enumerate(zip(train_loader, train_loader)):
        x1, y1 = x1.to(device), y1.to(device)
        x2, y2 = x2.to(device), y2.to(device)

        _, y1 = y1.max(dim=1)
        _, y2 = y2.max(dim=1)

        lam = np.random.beta(mix_alpha, mix_alpha, len(x1))
        lam = torch.from_numpy(lam)
        lam = lam.to(device, dtype=torch.float)
        lamx = lam.view(-1, 1, 1, 1)
        mix_x = x1 * lamx + x2 * (1. - lamx)
        logits = model(mix_x)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss = mixup_criterion(criterion, logits, y1, y2, lam)

        if sz_penalty > 0.0:
            loss = loss + sum_zero_loss(model, sz_penalty)

        loss_mean = torch.mean(loss)

        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()

def train_sda_mixup(src_loader, dst_loader, model, optimizer, device, mix_alpha=0.1):
    model.train(mode=True)
    for idx, ((x_src, y_src), (x_dst, y_dst)) in enumerate(zip(src_loader, dst_loader)):
        x_src, y_src = x_src.to(device), y_src.to(device)
        x_dst, y_dst = x_dst.to(device), y_dst.to(device)

        # onehot to int
        _, y_src = y_src.max(dim=1)
        _, y_dst = y_dst.max(dim=1)

        lam = np.random.beta(mix_alpha, mix_alpha, len(x_src))
        lam = torch.from_numpy(lam)
        # .to() will copy data
        lam = lam.to(device=device, dtype=torch.float)
        # share the same data
        lamx = lam.view(-1, 1, 1, 1)
        mix_x = x_src * lamx + x_dst * (1. - lamx)
        logits = model(mix_x)
        # inside CE: combined LogSoftmax and NLLLoss
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        loss = mixup_criterion(criterion, logits, y_src, y_dst, lam)

        loss_mean = torch.mean(loss)

        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()

def train_mixup(train_loader, model, optimizer, device):
    model.train(mode=True)
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # onehot to int
        _, y = y.max(dim=1)

        x, y1, y2, lam = mixup_data(x, y, alpha=1, use_cuda=True)

        logits = model(x)

        # inside CE: combined LogSoftmax and NLLLoss
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        loss = mixup_criterion(criterion, logits, y1, y2, lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_model(train_loader, model, optimizer, device):
    model.train(mode=True)
    train_loss = 0
    correct = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        _, y = y.max(dim=1)

        logits = model(x)

        # inside CE: combined LogSoftmax and NLLLoss
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        with torch.no_grad():
            pred = logits.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
    return {'loss': train_loss, 'acc': train_acc}


def eval_model(test_loader, model, device):

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, target = target.max(dim=1)
            logits = model(data)
            # inside CE: combined LogSoftmax and NLLLoss
            criterion = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = criterion(logits, target)

            test_loss += loss.item() # sum up batch loss
            pred = logits.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    # print('\nEpoch{},Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     epoch, test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    return {'loss': test_loss, 'acc': test_acc}