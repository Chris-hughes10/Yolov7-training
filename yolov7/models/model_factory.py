from yolov7.models.model_configs import get_yolov7_config

import math
import torch

MODEL_CONFIGS = {
    'yolov7': get_yolov7_config
}

def create_model_from_config(architecture, num_classes=80, anchors=None):
    m = MODEL_CONFIGS[architecture](num_classes=num_classes, anchors=anchors)
    return m
