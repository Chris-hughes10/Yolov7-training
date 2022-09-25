# copied from https://github.com/WongKinYiu/yolov7/blob/main/models/yolo.py

import logging
import sys

import math
import torch
from torch import nn

from yolov7.anchors import check_anchor_order
from yolov7.migrated.models.common import RepConv
from yolov7.models.core.layers import Conv
from yolov7.models.core.detection_heads import Detect, IDetect, IAuxDetect
from yolov7.models.model_configs import parse_model
from yolov7.models.model_factory import create_model_from_config

sys.path.append("/")  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
from yolov7.migrated.utils.torch_utils import (
    fuse_conv_and_bn,
)


class Model(nn.Module):
    def __init__(
            self, architecture="yolov7", ch=3, nc=80, anchors=None, pretrained=False
    ):  # model, input channels, number of classes
        super(Model, self).__init__()
        self.pretrained = pretrained
        self.ch = ch

        model_config = create_model_from_config(architecture, num_classes=nc, anchors=anchors)
        self.model, self.save_output_layer_idxs = parse_model(model_config, [ch])

        self.initialize()

    def initialize(self):
        # Build strides, anchors
        m = self.model[-1]  # Detect()
        s = 256  # 2x min stride
        if isinstance(m, Detect) or isinstance(m, IDetect):
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, self.ch, s, s))]
            )  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride

            if not self.pretrained:
                _initialize_biases(self.model)  # only run once
        if isinstance(m, IAuxDetect):
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, self.ch, s, s))[:4]]
            )  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            if not self.pretrained:
                _initialize_aux_biases(self.model)  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)


    def forward(self, x):
        intermediate_outputs = []  # outputs
        for m in self.model:
            if m.from_index != -1:  # if not from previous layer
                x = (
                    intermediate_outputs[m.from_index]
                    if isinstance(m.from_index, int)
                    else [x if j == -1 else intermediate_outputs[j] for j in m.from_index]
                )  # from earlier layers

            if not hasattr(self, "traced"):
                self.traced = False

            if self.traced:
                if (
                        isinstance(m, Detect)
                        or isinstance(m, IDetect)
                        or isinstance(m, IAuxDetect)
                ):
                    break

            x = m(x)  # run

            intermediate_outputs.append(x if m.attach_index in self.save_output_layer_idxs else None)  # save output
        return x

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, RepConv):
                # print(f" fuse_repvgg_block")
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, (IDetect, IAuxDetect)):
                m.fuse()
                m.forward = m.fuseforward
        return self


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

def _initialize_biases(
            model, cf=None
    ):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

def _initialize_aux_biases(
            model, cf=None
    ):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = model[-1]  # Detect() module
        for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b2 = mi2.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b2.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b2.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)