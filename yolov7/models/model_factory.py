import math

import torch
from pytorch_accelerated.utils import local_process_zero_first
from torch import nn

from yolov7.models.core.detection_heads import IDetect, Detect, IAuxDetect
from yolov7.models.core.initialization import (
    initialize_weights,
    initialize_biases,
    initialize_aux_biases,
)
from yolov7.models.model_configs import get_yolov7_config
from yolov7.models.yolo import Yolov7Model
from yolov7.utils import intersect_dicts

MODEL_CONFIGS = {"yolov7": get_yolov7_config}



@local_process_zero_first
def create_yolov7_model(
    architecture, num_classes=80, anchors=None, num_channels=3, pretrained=True
):
    config = MODEL_CONFIGS[architecture](
        num_classes=num_classes, anchors=anchors, num_channels=num_channels
    )

    model = Yolov7Model(model_config=config)

    if pretrained:
        state_dict_path = config["state_dict_path"]
        try:
            # load state dict
            state_dict = intersect_dicts(
                torch.hub.load_state_dict_from_url(state_dict_path, progress=False),
                model.state_dict(),
                exclude=["anchor"],
            )
            model.load_state_dict(state_dict, strict=False)
            print(
                f"Transferred {len(state_dict)}/{len(model.state_dict())} items from {state_dict_path}"
            )
        except Exception as e:
            print(f'Unable to load pretrained model weights from {state_dict_path}')
            print(e)

    else:
        initialize_weights(model)

        detection_head = model.model[-1]

        if config["aux_detection"]:
            initialize_aux_biases(detection_head)
        else:
            initialize_biases(detection_head)

    return model