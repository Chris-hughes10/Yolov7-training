import torch
import yaml

from yolov7.migrated.models.yolo import Model
from yolov7.migrated.utils.general import xywh2xyxy
from yolov7.migrated.utils.torch_utils import intersect_dicts


def create_yolov7_model(
    config="yolov7.yaml", state_dict_path=None, num_classes=80, num_channels=3, anchors=None
):
    with open(config, "r") as file:
        model_cfg = yaml.safe_load(file)

    model = Model(model_cfg, ch=num_channels, nc=num_classes, anchors=anchors)
    if state_dict_path is not None:
        state_dict = intersect_dicts(
            torch.load(state_dict_path), model.state_dict(), exclude=["anchor"]
        )  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        print(
            f"Transferred {len(state_dict)}/{len(model.state_dict())} items from {state_dict_path}"
        )

    model.nc = num_classes  # attach number of classes to model
    return model


def process_yolov7_outputs(model_outputs, conf_thres=0.2, max_detections=30000):
    model_outputs = model_outputs[0]
    num_classes = model_outputs.shape[2] - 5

    outputs = [torch.zeros((0, 6), device=model_outputs.device)] * model_outputs.shape[0]

    for image_idx, detections_for_image in enumerate(model_outputs):  # image index, image inference

        # filter by confidence
        detections_for_image = detections_for_image[detections_for_image[:, 4] >= conf_thres]

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
        # TODO replace this method
        box = xywh2xyxy(detections_for_image[:, :4])

        # best class only
        # j, most confident class index
        conf, class_idx = detections_for_image[:, 5:].max(1, keepdim=True)

        # filter by class confidence
        detections_for_image = torch.cat((box, conf, class_idx), 1)[conf.view(-1) > conf_thres]

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
