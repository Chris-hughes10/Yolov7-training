# Adapted from from https://github.com/WongKinYiu/yolov7/blob/main/models/yolo.py

import logging
from copy import deepcopy

import torch
import torchvision
from torch import nn

from yolov7.anchors import check_anchor_order
from yolov7.migrated.models.common import RepConv
from yolov7.models.config_builder import create_model_from_config
from yolov7.models.core.detection_heads import Detect, IDetect, IAuxDetect
from yolov7.models.core.layer_operations import fuse_conv_and_bn
from yolov7.models.core.layers import Conv

logger = logging.getLogger(__name__)


class Yolov7Model(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        self.num_channels = self.config["num_channels"]
        self.nc = self.config["nc"]
        self.stride = None
        self.traced = False

        self.model, self.save_output_layer_idxs = create_model_from_config(
            model_config=deepcopy(self.config),
        )
        self.initialize_anchors()

    def initialize_anchors(self):
        detection_head = self.model[-1]
        s = 256  # 2x min stride
        if isinstance(detection_head, Detect) or isinstance(detection_head, IDetect):
            detection_head.stride = torch.tensor(
                [
                    s / x.shape[-2]
                    for x in self.forward(torch.zeros(1, self.num_channels, s, s))
                ]
            )

        elif isinstance(detection_head, IAuxDetect):
            detection_head.stride = torch.tensor(
                [
                    s / x.shape[-2]
                    for x in self.forward(torch.zeros(1, self.num_channels, s, s))[:4]
                ]
            )

        detection_head.anchors /= detection_head.stride.view(-1, 1, 1)
        check_anchor_order(detection_head)
        self.stride = detection_head.stride

    def forward(self, x):
        intermediate_outputs = []
        for module_ in self.model:
            if module_.from_index != -1:
                # if input not from previous layer, get intermediate outputs
                x = (
                    intermediate_outputs[module_.from_index]
                    if isinstance(module_.from_index, int)
                    else [
                        x if j == -1 else intermediate_outputs[j]
                        for j in module_.from_index
                    ]
                )

            if self.traced:
                if (
                        isinstance(module_, Detect)
                        or isinstance(module_, IDetect)
                        or isinstance(module_, IAuxDetect)
                ):
                    break

            x = module_(x)  # run

            intermediate_outputs.append(
                x if module_.attach_index in self.save_output_layer_idxs else None
            )  # save output
        return x


def process_yolov7_outputs(model_outputs, conf_thres=0.2, max_detections=30000):
    model_outputs = model_outputs[0]
    num_classes = model_outputs.shape[2] - 5

    outputs = [torch.zeros((0, 6), device=model_outputs.device)] * model_outputs.shape[
        0
    ]

    for image_idx, detections_for_image in enumerate(
            model_outputs
    ):  # image index, image inference

        # filter by confidence
        detections_for_image = detections_for_image[
            detections_for_image[:, 4] >= conf_thres
            ]

        # If none remain process next image
        if not detections_for_image.shape[0]:
            continue

        if num_classes == 1:
            detections_for_image[:, 5:] = detections_for_image[
                                          :, 4:5
                                          ]  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
            # so there is no need to multiply.
        else:
            detections_for_image[:, 5:] *= detections_for_image[
                                           :, 4:5
                                           ]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = torchvision.ops.box_convert(detections_for_image[:, :4], "cxcywh", "xyxy")

        # best class only
        # j, most confident class index
        conf, class_idx = detections_for_image[:, 5:].max(1, keepdim=True)

        # filter by class confidence
        detections_for_image = torch.cat((box, conf, class_idx), 1)[
            conf.view(-1) > conf_thres
            ]

        # Check shape
        n = detections_for_image.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_detections:  # excess boxes
            detections_for_image = detections_for_image[
                detections_for_image[:, 4].argsort(descending=True)[:max_detections]
            ]  # sort by confidence

        outputs[image_idx] = detections_for_image

    return outputs
