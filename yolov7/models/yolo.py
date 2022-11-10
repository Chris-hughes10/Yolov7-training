# Adapted from from https://github.com/WongKinYiu/yolov7/blob/main/models/yolo.py

import logging
from copy import deepcopy
from typing import List

import torch
import torchvision
from torch import nn
from yolov7.loss import PredIdx, transform_model_outputs_into_predictions
from yolov7.models.config_builder import create_model_from_config


logger = logging.getLogger(__name__)


class Yolov7Model(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        self.num_channels = self.config["num_channels"]
        self.num_classes = self.config["num_classes"]

        self.model, self.save_output_layer_idxs = create_model_from_config(
            model_config=deepcopy(self.config),
        )
        self.initialize_anchors()

    @property
    def detection_head(self):
        return self.model[-1]

    def get_parameter_groups(self):
        conv_weights = {
            v.weight
            for k, v in self.model.named_modules()
            if (
                hasattr(v, "weight")
                and isinstance(v.weight, nn.Parameter)
                and not isinstance(v, nn.BatchNorm2d)
            )
        }

        other_params = [p for p in self.model.parameters() if p not in conv_weights]

        return {"conv_weights": list(conv_weights), "other_params": other_params}

    def forward(self, x):
        intermediate_outputs = []
        for module_ in self.model:
            if module_.from_index != -1:
                # if input not from previous layer, get intermediate outputs
                if isinstance(module_.from_index, int):
                    x = intermediate_outputs[module_.from_index]
                else:
                    x = [
                        x if idx == -1 else intermediate_outputs[idx]
                        for idx in module_.from_index
                    ]

            x = module_(x)

            intermediate_outputs.append(
                x if module_.attach_index in self.save_output_layer_idxs else None
            )
        return x

    def postprocess(
        self,
        fpn_heads_outputs: List[torch.Tensor],
        conf_thres: float=0.001,
        max_detections: int=30000,
        multiple_labels_per_box: bool=True,
    ) -> List[torch.Tensor]:
        """Convert FPN outputs into human-interpretable box predictions

        The outputted predictions are a list and each element corresponds to one image, in the
        same order they were passed to the model.

        Each element is a tensor with all the box predictions for that image. The dimensions of
        such tensor are Nx6 (x1 y1 x2 y2 conf class_idx), where N is the number of outputted boxes.

        - If not `multiple_labels_per_box`: Only one box per output, with class with higher conf.
        - Otherwise: Box duplicated for each class with conf above `conf_thres`.
        """
        preds = self._derive_preds(fpn_heads_outputs)
        formatted_preds = self._format_preds(
            preds, conf_thres, max_detections, multiple_labels_per_box
        )
        return formatted_preds

    def _derive_preds(self, fpn_heads_outputs):
        all_preds = []
        for layer_idx, fpn_head_outputs in enumerate(fpn_heads_outputs):
            batch_size, _, num_rows, num_cols, *_ = fpn_head_outputs.shape
            grid = self._make_grid(num_rows, num_cols).to(fpn_head_outputs.device)
            fpn_head_preds = transform_model_outputs_into_predictions(fpn_head_outputs)
            fpn_head_preds[
                ..., [PredIdx.CY, PredIdx.CX]
            ] += grid  # Grid corrections -> Grid coordinates
            fpn_head_preds[..., [PredIdx.CX, PredIdx.CY]] *= self.detection_head.strides[
                layer_idx
            ]  # -> Image coordinates
            # TODO: Probably can do it in a more standardized way
            fpn_head_preds[
                ..., [PredIdx.W, PredIdx.H]
            ] *= self.detection_head.anchor_grid[
                layer_idx
            ]  # Anchor box corrections -> Image coordinates
            fpn_head_preds[..., PredIdx.OBJ :].sigmoid_()
            all_preds.append(
                fpn_head_preds.view(batch_size, -1, self.detection_head.num_outputs)
            )
        return torch.cat(all_preds, 1)

    @staticmethod
    def _make_grid(num_rows, num_cols):
        """Create grid with two stacked matrixes, one with col idxs and the other with row idxs
        """
        meshgrid = torch.meshgrid(
            [torch.arange(num_rows), torch.arange(num_cols)], indexing="ij"
        )
        grid = torch.stack(meshgrid, 2).view((1, 1, num_rows, num_cols, 2)).float()
        return grid

    def _format_preds(
        self,
        preds,
        conf_thres=0.001,
        max_detections=30000,
        multiple_labels_per_box=True,
    ):
        num_classes = preds.shape[2] - 5

        formatted_preds = [torch.zeros((0, 6), device=preds.device)] * preds.shape[0]

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
                    (
                        xyxy_boxes[box_idxs],
                        class_confidences,
                        class_idxs[:, None].float(),
                    ),
                    1,
                )

            else:
                # best class only
                # j, most confident class index
                class_conf, class_idxs = detections_for_image[:, 5:].max(
                    1, keepdim=True
                )

                # filter by class confidence
                detections_for_image = torch.cat(
                    (xyxy_boxes, class_conf, class_idxs), 1
                )[class_conf.view(-1) > conf_thres]

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
