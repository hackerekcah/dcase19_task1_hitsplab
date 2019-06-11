""" 
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
import numpy as np

# __all__ = ['xception']

model_urls = {
    'xception': 'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000, decay=0.9998):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        # self.fc = nn.Linear(2048, num_classes)

        # global weighted rank pooling
        self.gwrp1 = GWRP(decay=decay)
        self.gwrp2 = GWRP(decay=decay)
        self.gwrp3 = GWRP(decay=decay)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x11 = self.block11(x)
        x12 = self.block12(x11)

        x = self.conv3(x12)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.gwrp1(x)
        x = x.view(x.size(0), -1)

        # x11 = F.adaptive_avg_pool2d(x11, (1, 1))
        x11 = self.gwrp2(x11)
        x11 = x11.view(x11.size(0), -1)

        # x12 = F.adaptive_avg_pool2d(x12, (1, 1))
        x12 = self.gwrp3(x12)
        x12 = x12.view(x12.size(0), -1)

        return torch.cat((x11, x12, x), 1)


def xception(pretrained=False, **kwargs):
    """
    Construct Xception.
    """

    model = Xception(**kwargs)
    if pretrained:
        # removed fc from original xception, set strict=False to allow state key mismatch
        model.load_state_dict(model_zoo.load_url(model_urls['xception']), strict=False)
    return model


class ModifiedXception(nn.Module):
    def __init__(self, num_classes=10, drop_rate=0.3, decay=0.998):
        super(ModifiedXception, self).__init__()
        self.num_class = num_classes
        kwargs = {
            'decay': decay
        }
        self.xception = xception(pretrained=True, **kwargs)
        self.drop1 = nn.Dropout(p=drop_rate)
        self.fc1 = nn.Linear(in_features=2048+1024+728, out_features=512)
        self.drop2 = nn.Dropout(p=drop_rate)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        # remove fc from original xception
        x = self.xception(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x


class XceptionSpecMean(nn.Module):
    def __init__(self, num_classes=10, drop_rate=0.3, decay=0.998):
        super(XceptionSpecMean, self).__init__()
        self.num_class = num_classes
        kwargs = {
            'decay': decay
        }
        self.xception = xception(pretrained=True, **kwargs)
        self.drop1 = nn.Dropout(p=drop_rate)
        self.fc1 = nn.Linear(in_features=2048+1024+728+128, out_features=512)
        self.drop2 = nn.Dropout(p=drop_rate)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        # get mean value
        # (Batch,Channel,Height,Width) -> (Batch, Height)
        spec_mean = torch.mean(torch.mean(x, dim=1), dim=-1)

        # subtract mean, so that xcept learns the discriminative feature
        x = x - torch.mean(x, dim=-1, keepdim=True)

        # remove fc from original xception
        x1 = self.xception(x)

        # get the mean info back
        x = torch.cat((x1, spec_mean), 1)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x


class GWRP(nn.Module):
    def __init__(self, decay=0.9998):
        """
        Global Weighted Rank Pooling
        :param decay:
        :param cuda: using cuda or not
        """
        super(GWRP, self).__init__()
        self.decay = torch.nn.Parameter(data=torch.tensor(decay, dtype=torch.float), requires_grad=False)

    def forward(self, input):
        """
        forward function
        :param input: (batch, channel, Height, Width)
        :return:
        """
        gwrp_w = self.decay ** torch.arange(input.shape[2] * input.shape[3]).type_as(self.decay)
        sum_gwrp_w = torch.sum(gwrp_w)

        x = input.view(input.shape[0], input.shape[1], input.shape[2] * input.shape[3])
        """(batch_size, channel, freq*time_seq)"""
        x, _ = torch.sort(x, dim=-1, descending=True)

        x = x * gwrp_w[None, None, :] # be equal to x * gwrp_w, (batch_size, channel, freq * time_seq)
        x = torch.sum(x, dim=-1) # (batch_size, channel)

        output = x / sum_gwrp_w
        return output


if __name__ == '__main__':
    model = ModifiedXception()
    params = list(model.parameters())
    total_params = 0
    for i in params:
        l = 1
        print('layer structure: ', str(list(i.size())))
        for j in i.size():
            l *= j
        print('layer parameters: ', str(l))
        total_params += l
    print('total parameters: ', str(total_params))
    # out = model(torch.randn(1, 3, 128, 430))
    # print(out.shape)
