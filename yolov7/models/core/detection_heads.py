from typing import List
import torch
from torch import nn

from yolov7.models.core.layers import ImplicitAdd, ImplicitMultiply


class Yolov7DetectionHead(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        anchor_sizes_per_layer: torch.Tensor = (),
        strides: torch.Tensor = (),
        in_channels_per_layer: List[int] = (),
    ):
        """
        :param num_classes:
        :param anchor_sizes_per_layer: (num_layers, num_anchor_sizes, 2)
        :param strides: (num_layers)
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = 5 + num_classes  # xywh + obj + cls1 + cls2 + ...
        self.num_layers = anchor_sizes_per_layer.shape[0]
        self.num_anchor_sizes = anchor_sizes_per_layer.shape[1]
        self.grid = [torch.zeros(1)] * self.num_layers  # init grid
        self.strides = strides
        self.register_buffer(
            "anchor_grid",
            anchor_sizes_per_layer.float().view(
                self.num_layers, 1, self.num_anchor_sizes, 1, 1, 2
            ),
        )
        grid_coord_anchor_sizes = anchor_sizes_per_layer.float() / strides.view(
            -1, 1, 1
        )
        self.register_buffer("anchor_sizes_per_layer", grid_coord_anchor_sizes)

        # With [:self.num_layers], ensure we only use lead heads and no aux heads.
        self.m = nn.ModuleList(
            nn.Conv2d(in_channels, self.num_outputs * self.num_anchor_sizes, 1)
            for in_channels in in_channels_per_layer[: self.num_layers]
        )  # _conv2d_list
        self.ia = nn.ModuleList(
            ImplicitAdd(in_channels)
            for in_channels in in_channels_per_layer[: self.num_layers]
        )  # _impl_add_list
        self.im = nn.ModuleList(
            ImplicitMultiply(self.num_outputs * self.num_anchor_sizes)
            for _ in in_channels_per_layer[: self.num_layers]
        )  # _impl_mult_list

    def forward(self, x):
        for layer_idx in range(self.num_layers):
            self.layer_forward(x, layer_idx)
        return x

    def layer_forward(self, x, layer_idx):
        x[layer_idx] = self.m[layer_idx](self.ia[layer_idx](x[layer_idx]))
        x[layer_idx] = self.im[layer_idx](x[layer_idx])

        batch_size, _, grid_rows, grid_cols = x[layer_idx].shape
        x[layer_idx] = (
            x[layer_idx]
            .view(
                batch_size,
                self.num_anchor_sizes,
                self.num_outputs,
                grid_rows,
                grid_cols,
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )


class Yolov7DetectionHeadWithAux(Yolov7DetectionHead):
    def __init__(
        self,
        num_classes=80,
        anchor_sizes_per_layer=(),
        strides: torch.Tensor = (),
        in_channels_per_layer=(),
    ):
        super().__init__(
            num_classes, anchor_sizes_per_layer, strides, in_channels_per_layer
        )
        self.m2 = nn.ModuleList(
            nn.Conv2d(in_channels, self.num_outputs * self.num_anchor_sizes, 1)
            for in_channels in in_channels_per_layer[self.num_layers :]
        )  # _aux_module_list

    def aux_layer_forward(self, x, layer_idx):
        batch_size, _, grid_rows, grid_cols, _ = x[layer_idx].shape
        x[layer_idx + self.num_layers] = self.m2[layer_idx](
            x[layer_idx + self.num_layers]
        )
        x[layer_idx + self.num_layers] = (
            x[layer_idx + self.num_layers]
            .view(
                batch_size,
                self.num_anchor_sizes,
                self.num_outputs,
                grid_rows,
                grid_cols,
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

    def forward(self, x):
        for i in range(self.num_layers):
            self.layer_forward(x, i)
            self.aux_layer_forward(x, i)
        return x
