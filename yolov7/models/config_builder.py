import math

from torch import nn
from yolov7.models.core.detection_heads import (
    Yolov7DetectionHead,
    Yolov7DetectionHeadWithAux,
)
from yolov7.models.core.layers import (
    SPPCSPC,
    Concat,
    Conv,
    DownC,
    ReOrg,
    RepConv,
    Shortcut,
)


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def create_model_from_config(model_config):
    ch = [model_config["num_channels"]]
    anchor_sizes_per_layer = model_config["anchor_sizes_per_layer"]
    num_classes = model_config["num_classes"]
    gd = model_config["depth_multiple"]
    gw = model_config["width_multiple"]

    num_anchor_sizes = anchor_sizes_per_layer.shape[1]
    num_outputs = num_anchor_sizes * (num_classes + 5)
    layers = []
    save_output_layer_idxs = []
    num_out_channels = ch[-1]

    full_config = model_config["backbone"] + model_config["head"]

    for attach_idx, (from_idx, num_repeats, module_, module_args) in enumerate(
        full_config
    ):

        num_repeats = (
            max(round(num_repeats * gd), 1) if num_repeats > 1 else num_repeats
        )  # depth gain

        if module_.__name__ in {
            nn.Conv2d.__name__,
            Conv.__name__,
            RepConv.__name__,
            DownC.__name__,
            SPPCSPC.__name__,
        }:
            num_in_channels = ch[from_idx]
            num_out_channels = module_args[0]

            if num_out_channels != num_outputs:
                # if not output
                num_out_channels = make_divisible(num_out_channels * gw, 8)

            module_args = [num_in_channels, num_out_channels, *module_args[1:]]

            if module_.__name__ in {
                DownC.__name__,
                SPPCSPC.__name__,
            }:
                module_args.insert(2, num_repeats)
                num_repeats = 1

        elif module_.__name__ == nn.BatchNorm2d.__name__:
            module_args = [ch[from_idx]]
        elif module_.__name__ == Concat.__name__:
            num_out_channels = sum([ch[x] for x in from_idx])
        elif module_.__name__ == Shortcut.__name__:
            num_out_channels = ch[from_idx[0]]
        elif module_.__name__ in {
            Yolov7DetectionHead.__name__,
            Yolov7DetectionHeadWithAux.__name__,
        }:
            module_args.append([ch[x] for x in from_idx])
        elif module_ is ReOrg:
            num_out_channels = ch[from_idx] * 4
        else:
            num_out_channels = ch[from_idx]

        m_ = _create_module(module_, module_args, num_repeats, attach_idx, from_idx)

        save_output_layer_idxs.extend(
            x % attach_idx
            for x in ([from_idx] if isinstance(from_idx, int) else from_idx)
            if x != -1
        )
        layers.append(m_)
        if attach_idx == 0:
            ch = []
        ch.append(num_out_channels)
    return nn.Sequential(*layers), sorted(save_output_layer_idxs)


def _create_module(module_, module_args, num_repeats, attach_index, from_idx):
    m_ = (
        nn.Sequential(*[module_(*module_args) for _ in range(num_repeats)])
        if num_repeats > 1
        else module_(*module_args)
    )
    module_type = str(module_)[8:-2].replace("__main__.", "")
    num_params = sum([x.numel() for x in m_.parameters()])

    m_.attach_index = attach_index
    m_.from_index = from_idx
    m_.module_type = module_type
    m_.num_params = num_params

    return m_
