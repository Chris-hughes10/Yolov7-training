import torch
from torch import nn

from yolov7.models.core.layers import ImplicitAdd, ImplicitMultiply


class Yolov7DetectionHead(nn.Module):

    def __init__(self, num_classes=80, anchor_sizes=(), in_channels_per_layer=()):  # detection layer
        super().__init__()
        self.num_classes = num_classes  # number of classes
        self.num_outputs = 5 + num_classes  # xywh + obj + cls1 + cls2 + ...
        self.num_layers = len(anchor_sizes)  # number of detection layers
        self.num_anchor_sizes = len(anchor_sizes[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.num_layers  # init grid
        a = torch.tensor(anchor_sizes).float().view(self.num_layers, self.num_anchor_sizes, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.num_layers, 1, self.num_anchor_sizes, 1, 1, 2)
        )
        self.initialize_module_parameters(in_channels_per_layer)

    def initialize_module_parameters(self, in_channels_per_layer):
        self._conv2d_list = nn.ModuleList(
            nn.Conv2d(in_channels, self.num_outputs * self.num_anchor_sizes, 1)
            for in_channels in in_channels_per_layer
        )
        self._impl_add_list = nn.ModuleList(
            ImplicitAdd(in_channels) for in_channels in in_channels_per_layer
        )
        self._impl_mult_list = nn.ModuleList(
            ImplicitMultiply(self.num_outputs * self.num_anchor_sizes) for _ in in_channels_per_layer
        )

    def forward(self, x):
        for layer_idx in range(self.num_layers):
            self.layer_forward(x, layer_idx)
        return x

    def layer_forward(self, x, layer_idx):
        x[layer_idx] = self._conv2d_list[layer_idx](
            self._impl_add_list[layer_idx](x[layer_idx])
        )
        x[layer_idx]  = self._impl_mult_list[layer_idx](x[layer_idx])

        batch_size, _, grid_rows, grid_cols = x[layer_idx].shape
        x[layer_idx] = (
            x[layer_idx]
            .view(batch_size, self.num_anchor_sizes, self.num_outputs, grid_rows, grid_cols)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )


class Yolov7DetectionHeadWithAux(Yolov7DetectionHead):
    def initialize_module_parameters(self, in_channels_per_layer):
        super().initialize_module_parameters(in_channels_per_layer[:self.num_layers])
        self._aux_module_list = nn.ModuleList(
            nn.Conv2d(in_channels, self.num_outputs * self.num_anchor_sizes, 1)
            for in_channels in in_channels_per_layer[self.num_layers:]
        )


    def aux_layer_forward(self, x, layer_idx):
        batch_size, _, grid_rows, grid_cols, _ = x[layer_idx].shape
        x[layer_idx + self.num_layers] = self._aux_module_list[layer_idx](x[layer_idx + self.num_layers])
        x[layer_idx + self.num_layers] = (
            x[layer_idx + self.num_layers]
            .view(batch_size, self.num_anchor_sizes, self.num_outputs, grid_rows, grid_cols)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

    def forward(self, x):
        for i in range(self.num_layers):
            self.layer_forward(x, i)
            self.aux_layer_forward(x, i)
        return x
