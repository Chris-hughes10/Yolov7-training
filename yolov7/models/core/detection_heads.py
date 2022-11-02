import torch
from torch import nn

from yolov7.models.core.layers import ImplicitAdd, ImplicitMultiply


class Yolov7DetectionHead(nn.Module):
    def __init__(self, num_classes=80, anchors=(), ch=()):  # detection layer
        super().__init__()
        self.num_classes = num_classes  # number of classes
        self.num_outputs = num_classes + 5  # number of outputs per anchor
        self.num_detection_layers = len(anchors)  # number of detection layers
        self.num_anchor_templates = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.num_detection_layers  # init grid
        anchors = torch.tensor(anchors).float().view(self.num_detection_layers, -1, 2)
        self.register_buffer("anchors", anchors)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid",
            anchors.clone().view(self.num_detection_layers, 1, -1, 1, 1, 2),
        )  # shape(nl,1,na,1,1,2)
        self._module_list = None
        self.ia = None
        self.im = None
        self.initialize_module_parameters(ch)

    def initialize_module_parameters(self, ch):
        self._module_list = nn.ModuleList(
            nn.Conv2d(x, self.num_outputs * self.num_anchor_templates, 1) for x in ch
        )  # output conv

        self.ia = nn.ModuleList(ImplicitAdd(x) for x in ch)
        self.im = nn.ModuleList(
            ImplicitMultiply(self.num_outputs * self.num_anchor_templates) for _ in ch
        )

    def forward(self, x):
        for i in range(self.num_detection_layers):
            self.layer_forward(x, i)
        return x

    def layer_forward(self, x, i):
        x[i] = self._module_list[i](self.ia[i](x[i]))  # conv
        x[i] = self.im[i](x[i])
        batch_size, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x[i] = (
            x[i]
            .view(batch_size, self.num_anchor_templates, self.num_outputs, ny, nx)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )


class Yolov7DetectionHeadWithAux(Yolov7DetectionHead):
    def initialize_module_parameters(self, ch):
        super().initialize_module_parameters(ch[: self.num_detection_layers])
        self._aux_module_list = nn.ModuleList(
            nn.Conv2d(x, self.num_outputs * self.num_anchor_templates, 1)
            for x in ch[self.num_detection_layers :]
        )  # output conv

    def aux_layer_forward(self, x, i):
        batch_size, _, ny, nx = x[i].shape
        x[i + self.num_detection_layers] = self._aux_module_list[i](
            x[i + self.num_detection_layers]
        )
        x[i + self.num_detection_layers] = (
            x[i + self.num_detection_layers]
            .view(batch_size, self.num_anchor_templates, self.num_outputs, ny, nx)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

    def forward(self, x):
        for i in range(self.num_detection_layers):
            self.layer_forward(x, i)
            self.aux_layer_forward(x, i)
        return x
