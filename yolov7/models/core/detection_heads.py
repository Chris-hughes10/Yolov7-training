import torch
from torch import nn

from yolov7.models.core.layers import ImplicitAdd, ImplicitMultiply


class Yolov7DetectionHead(nn.Module):

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        # TODO: Nasty that not everything is defined here
        self.initialize_module_parameters(self, ch)

    def initialize_module_parameters(self, ch):
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv

        self.ia = nn.ModuleList(ImplicitAdd(x) for x in ch)
        self.im = nn.ModuleList(ImplicitMultiply(self.no * self.na) for _ in ch)

    def forward(self, x):
        for i in range(self.nl):
            self.layer_forward(x, i)
        return x

    def layer_forward(self, x, i):
        x[i] = self.m[i](self.ia[i](x[i]))  # conv
        x[i] = self.im[i](x[i])
        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x[i] = (
            x[i]
            .view(bs, self.na, self.no, ny, nx)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )


class Yolov7DetectionHeadWithAux(Yolov7DetectionHead):
    def initialize_module_parameters(self, ch):
        super().initialize_module_parameters(ch[:self.nl])
        self.m2 = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch[self.nl :]
        )  # output conv

    def aux_layer_forward(self, x, i):
        bs, _, ny, nx = x[i].shape
        x[i + self.nl] = self.m2[i](x[i + self.nl])
        x[i + self.nl] = (
            x[i + self.nl]
            .view(bs, self.na, self.no, ny, nx)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

    def forward(self, x):
        for i in range(self.nl):
            self.layer_forward(x, i)
            self.aux_layer_forward(x, i)
        return x


