import math

import torch
from torch import nn


def initialize_weights(model):

    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def initialize_biases(
    detection_head, cf=None
):  # initialize biases into Detect(), cf is class frequency
    # https://arxiv.org/abs/1708.02002 section 3.3
    # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
    m = detection_head  # Detect() module
    for mi, s in zip(m.m, m.stride):  # from
        b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
        b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
        b.data[:, 5:] += (
            math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
        )  # cls
        mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def initialize_aux_biases(
    detection_head, cf=None
):  # initialize biases into Detect(), cf is class frequency
    # https://arxiv.org/abs/1708.02002 section 3.3
    # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
    # m = model.model[-1]  # Detect() module
    m = detection_head
    for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
        b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
        b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
        b.data[:, 5:] += (
            math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
        )  # cls
        mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        b2 = mi2.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
        b2.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
        b2.data[:, 5:] += (
            math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
        )  # cls
        mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)
