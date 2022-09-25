from torch import nn

from yolov7.models.core.layers import Conv, Concat, MP, SPPCSPC, RepConv, DownC, Shortcut, ReOrg
from yolov7.models.core.detection_heads import IDetect, Detect, IAuxDetect
from yolov7.migrated_refactor.utils.general import make_divisible


def get_yolov7_config(num_classes=80, anchors=None):
    if anchors is None:
        anchors = [
            [12, 16, 19, 36, 40, 28],
            [36, 75, 76, 55, 72, 146],
            [142, 110, 192, 243, 459, 401],
        ]

    # TODO detect module
        # if training IDetect, other Detect

    return {
        "nc": num_classes,
        "image_size": (640, 640),
        "depth_multiple": 1.0,
        "width_multiple": 1.0,
        "anchors": anchors,
        "backbone": [
            [-1, 1, Conv, [32, 3, 1]],
            [-1, 1, Conv, [64, 3, 2]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [128, 3, 2]],
            [-1, 1, Conv, [64, 1, 1]],
            [-2, 1, Conv, [64, 1, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [[-1, -3, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],
            [-1, 1, MP, []],
            [-1, 1, Conv, [128, 1, 1]],
            [-3, 1, Conv, [128, 1, 1]],
            [-1, 1, Conv, [128, 3, 2]],
            [[-1, -3], 1, Concat, [1]],
            [-1, 1, Conv, [128, 1, 1]],
            [-2, 1, Conv, [128, 1, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [[-1, -3, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [512, 1, 1]],
            [-1, 1, MP, []],
            [-1, 1, Conv, [256, 1, 1]],
            [-3, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [256, 3, 2]],
            [[-1, -3], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],
            [-2, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [[-1, -3, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [1024, 1, 1]],
            [-1, 1, MP, []],
            [-1, 1, Conv, [512, 1, 1]],
            [-3, 1, Conv, [512, 1, 1]],
            [-1, 1, Conv, [512, 3, 2]],
            [[-1, -3], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],
            [-2, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [[-1, -3, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [1024, 1, 1]],
        ],
        "head": [
            [-1, 1, SPPCSPC, [512]],
            [-1, 1, Conv, [256, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, "nearest"]],
            [37, 1, Conv, [256, 1, 1]],
            [[-1, -2], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],
            [-2, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [128, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, "nearest"]],
            [24, 1, Conv, [128, 1, 1]],
            [[-1, -2], 1, Concat, [1]],
            [-1, 1, Conv, [128, 1, 1]],
            [-2, 1, Conv, [128, 1, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [-1, 1, Conv, [64, 3, 1]],
            [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [128, 1, 1]],
            [-1, 1, MP, []],
            [-1, 1, Conv, [128, 1, 1]],
            [-3, 1, Conv, [128, 1, 1]],
            [-1, 1, Conv, [128, 3, 2]],
            [[-1, -3, 63], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],
            [-2, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [-1, 1, Conv, [128, 3, 1]],
            [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [256, 1, 1]],
            [-1, 1, MP, []],
            [-1, 1, Conv, [256, 1, 1]],
            [-3, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [256, 3, 2]],
            [[-1, -3, 51], 1, Concat, [1]],
            [-1, 1, Conv, [512, 1, 1]],
            [-2, 1, Conv, [512, 1, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [-1, 1, Conv, [256, 3, 1]],
            [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
            [-1, 1, Conv, [512, 1, 1]],
            [75, 1, RepConv, [256, 3, 1]],
            [88, 1, RepConv, [512, 3, 1]],
            [101, 1, RepConv, [1024, 3, 1]],
            [[102, 103, 104], 1, IDetect, [num_classes, anchors]],
        ],
    }


def parse_model(model_config, ch):  # model_dict, input_channels(3)
    print(
        "\n%3s%18s%3s%10s  %-40s%-30s"
        % ("", "from", "n", "params", "module", "arguments")
    )
    anchors = model_config["anchors"]
    num_classes = model_config["nc"]
    gd = model_config["depth_multiple"]
    gw = model_config["width_multiple"]

    num_anchors = (
        (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    )  # number of anchors
    num_outputs = num_anchors * (num_classes + 5)  # anchors * (classes + 5)
    layers = []
    save = []  # savelist
    num_out_channels = ch[-1]  # ch out

    full_config = model_config["backbone"] + model_config["head"]

    for attach_idx, (from_idx, num_repeats, module_, module_args) in enumerate(
            full_config
    ):  # from, number, module, args

        # module_, module_args = _parse_module_and_args(module_, module_args)

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

            if num_out_channels != num_outputs:  # if not output
                num_out_channels = make_divisible(num_out_channels * gw, 8)

            module_args = [num_in_channels, num_out_channels, *module_args[1:]]

            if module_.__name__ in {
                DownC.__name__,
                SPPCSPC.__name__,
            }:
                module_args.insert(2, num_repeats)  # number of repeats
                num_repeats = 1

        elif module_.__name__ == nn.BatchNorm2d.__name__:
            module_args = [ch[from_idx]]
        elif module_.__name__ == Concat.__name__:
            num_out_channels = sum([ch[x] for x in from_idx])
        elif module_.__name__ == Shortcut.__name__:
            num_out_channels = ch[from_idx[0]]
        elif module_.__name__ in {Detect.__name__, IDetect.__name__, IAuxDetect.__name__}:
            module_args.append([ch[x] for x in from_idx])
            if isinstance(module_args[1], int):  # number of anchors
                module_args[1] = [list(range(module_args[1] * 2))] * len(from_idx)
        elif module_ is ReOrg:
            num_out_channels = ch[from_idx] * 4
        else:
            num_out_channels = ch[from_idx]

        m_ = _create_module(module_, module_args, num_repeats, attach_idx, from_idx)

        save.extend(
            x % attach_idx
            for x in ([from_idx] if isinstance(from_idx, int) else from_idx)
            if x != -1
        )  # append to savelist
        layers.append(m_)
        if attach_idx == 0:
            ch = []
        ch.append(num_out_channels)
    return nn.Sequential(*layers), sorted(save)

def _create_module(module_, module_args, num_repeats, attach_index, from_idx):
    m_ = (
        nn.Sequential(*[module_(*module_args) for _ in range(num_repeats)])
        if num_repeats > 1
        else module_(*module_args)
    )  # module
    module_type = str(module_)[8:-2].replace("__main__.", "")  # module type
    num_params = sum([x.numel() for x in m_.parameters()])  # number params

    m_.attach_index = attach_index  # i
    m_.from_index = from_idx  # f
    m_.module_type = module_type # type
    m_.num_params = num_params # np

    print(
        "%3s%18s%3s%10.0f  %-40s%-30s"
        % (attach_index, from_idx, num_repeats, num_params, module_type, module_args)
    )  # print

    return m_