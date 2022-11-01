# Adapted from from https://github.com/WongKinYiu/yolov7/blob/main/models/yolo.py

import logging
from copy import deepcopy

import torch
import torchvision
from torch import nn

from yolov7.anchors import check_anchor_order
from yolov7.models.config_builder import create_model_from_config
from yolov7.models.core.detection_heads import Yolov7DetectionHead, Yolov7DetectionHeadWithAux, Detect, IDetect, IAuxDetect
from yolov7.loss import transform_model_outputs_into_predictions, PredIdx

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

    @property
    def detection_head(self):
        return self.model[-1]

    def initialize_anchors(self):
        detection_head = self.model[-1]
        s = 256  # 2x min stride
        if isinstance(detection_head, Yolov7DetectionHead):
            detection_head.stride = torch.tensor(
                [
                    s / x.shape[-2]
                    for x in self.forward(torch.zeros(1, self.num_channels, s, s))
                ]
            )

        elif isinstance(detection_head, Yolov7DetectionHeadWithAux):
            detection_head.stride = torch.tensor(
                [
                    s / x.shape[-2]
                    for x in self.forward(torch.zeros(1, self.num_channels, s, s))[:4]
                ]
            )

        detection_head.anchors /= detection_head.stride.view(-1, 1, 1) # Anchors into grid coordinates
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

            x = module_(x)  # run

            intermediate_outputs.append(
                x if module_.attach_index in self.save_output_layer_idxs else None
            )  # save output
        return x

    def postprocess(self, fpn_heads_outputs, conf_thres=0.001, max_detections=30000, multiple_labels_per_box=True):
        """TODO: Docstring"""
        # never ran in training
        # list of each detection head output (which changes grid x and grid y and anchors)
        # num_images, num_anchors, grid_x, grid_y, 4+1+num_classes
        preds = self._derive_preds(fpn_heads_outputs)
        formatted_preds = self._format_preds(preds, conf_thres, max_detections, multiple_labels_per_box)
        return formatted_preds

    def _derive_preds(self, fpn_heads_outputs):
        all_preds = []
        for layer_idx, fpn_head_outputs in enumerate(fpn_heads_outputs):
            batch_size, _, num_rows, num_cols, *_ = fpn_head_outputs.shape
            grid = self._make_grid(num_rows, num_cols).to(fpn_head_outputs.device)
            fpn_head_preds = transform_model_outputs_into_predictions(fpn_head_outputs)
            fpn_head_preds[..., [PredIdx.CY, PredIdx.CX]] += grid  # Grid corrections -> Grid coordinates
            fpn_head_preds[..., [PredIdx.CX, PredIdx.CY]] *= self.detection_head.stride[layer_idx]  # -> Image coordinates
            # TODO: Probably can do it in a more standardized way
            fpn_head_preds[..., [PredIdx.W, PredIdx.H]] *= self.detection_head.anchor_grid[layer_idx] # Anchor box corrections -> Image coordinates
            fpn_head_preds[..., PredIdx.OBJ:].sigmoid_()
            # TODO: Check if view is needed
            all_preds.append(fpn_head_preds.view(batch_size, -1, self.detection_head.no))
            # TODO: Before there was a .view(bs, -1, self.detection_head.no) in preds, check it
        return torch.cat(all_preds, 1)


    @staticmethod
    def _make_grid(num_rows, num_cols):
        """Create grid with two stacked matrixes, one with col idxs and the other with row idxs

        # TODO: Add example of matrixes
        """
        meshgrid = torch.meshgrid([torch.arange(num_rows), torch.arange(num_cols)], indexing="ij")
        grid = torch.stack(meshgrid, 2).view((1, 1, num_rows, num_cols, 2)).float()
        return grid

    def _format_preds(self, preds, conf_thres=0.001, max_detections=30000, multiple_labels_per_box=True):
        num_classes = preds.shape[2] - 5

        formatted_preds = [torch.zeros((0, 6), device=preds.device)] * preds.shape[
            0
        ]

        for image_idx, detections_for_image in enumerate(
            preds
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

            # Box non-normalized (center x, center y, width, height) to (x1, y1, x2, y2)
            xyxy_boxes = torchvision.ops.box_convert(
                detections_for_image[:, :4], "cxcywh", "xyxy"
            )

            if multiple_labels_per_box:
                # Detections matrix nx6 (xyxy, conf, cls)
                # keep multiple labels per box
                box_idxs, class_idxs = (
                    (detections_for_image[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                )
                class_confidences = detections_for_image[box_idxs, class_idxs + 5, None]
                detections_for_image = torch.cat(
                    (xyxy_boxes[box_idxs], class_confidences, class_idxs[:, None].float()),
                    1,
                )

            else:
                # best class only
                # j, most confident class index
                class_conf, class_idxs = detections_for_image[:, 5:].max(1, keepdim=True)

                # filter by class confidence
                detections_for_image = torch.cat((xyxy_boxes, class_conf, class_idxs), 1)[
                    class_conf.view(-1) > conf_thres
                ]

            # Check shape
            n = detections_for_image.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_detections:  # excess boxes
                detections_for_image = detections_for_image[
                    detections_for_image[:, 4].argsort(descending=True)[:max_detections]
                ]  # sort by confidence

            formatted_preds[image_idx] = detections_for_image

        return formatted_preds


def scale_bboxes_to_original_image_size(
    xyxy_boxes, resized_hw, original_hw, is_padded=True
):
    scaled_boxes = xyxy_boxes.clone()
    scale_ratio = resized_hw[0] / original_hw[0], resized_hw[1] / original_hw[1]

    if is_padded:
        # remove padding
        pad_scale = min(scale_ratio)
        padding = (resized_hw[1] - original_hw[1] * pad_scale) / 2, (
            resized_hw[0] - original_hw[0] * pad_scale
        ) / 2
        scaled_boxes[:, [0, 2]] -= padding[0]  # x padding
        scaled_boxes[:, [1, 3]] -= padding[1]  # y padding
        scale_ratio = (pad_scale, pad_scale)

    scaled_boxes[:, [0, 2]] /= scale_ratio[1]
    scaled_boxes[:, [1, 3]] /= scale_ratio[0]

    # Clip bounding xyxy bounding boxes to image shape (height, width)
    scaled_boxes[:, 0].clamp_(0, original_hw[1])  # x1
    scaled_boxes[:, 1].clamp_(0, original_hw[0])  # y1
    scaled_boxes[:, 2].clamp_(0, original_hw[1])  # x2
    scaled_boxes[:, 3].clamp_(0, original_hw[0])  # y2

    return scaled_boxes


#################
# LEGACY ########
#################

class LegacyYolov7Model(nn.Module):
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


def legacy_process_yolov7_outputs(
    model_outputs, conf_thres=0.001, max_detections=30000, multiple_labels_per_box=True
):
    # TODO move this function inside the model
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

        # Box non-normalized (center x, center y, width, height) to (x1, y1, x2, y2)
        xyxy_boxes = torchvision.ops.box_convert(
            detections_for_image[:, :4], "cxcywh", "xyxy"
        )

        if multiple_labels_per_box:
            # Detections matrix nx6 (xyxy, conf, cls)
            # keep multiple labels per box
            box_idxs, class_idxs = (
                (detections_for_image[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            )
            class_confidences = detections_for_image[box_idxs, class_idxs + 5, None]
            detections_for_image = torch.cat(
                (xyxy_boxes[box_idxs], class_confidences, class_idxs[:, None].float()),
                1,
            )

        else:
            # best class only
            # j, most confident class index
            class_conf, class_idxs = detections_for_image[:, 5:].max(1, keepdim=True)

            # filter by class confidence
            detections_for_image = torch.cat((xyxy_boxes, class_conf, class_idxs), 1)[
                class_conf.view(-1) > conf_thres
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
