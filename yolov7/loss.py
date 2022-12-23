# adapted from https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py
# Loss functions
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ORIG_IMAGE_SIZE = 640  # Image size used for original training of Yolov7
COCO_NUM_CLASSES = 80  # Original model was trained on COCO MS dataset


class TargetIdx:
    """Provide names for the each tensor idx for what constitutes a target"""

    IMG_IDX = 0  # Image index in batch
    CLS_ID = 1
    CX = 2
    CY = 3
    W = 4
    H = 5


class PredIdx:
    """Provide names for each tensor idx for what constitutes a prediction"""

    CX = 0
    CY = 1
    W = 2
    H = 3
    OBJ = 4
    # 5 to end correspond to class preds


class AnchorIdx:
    """Provide names for each tensor idx for what constitutes and anchor box"""

    IMG_IDX = 0  # Image index in batch
    ANCHOR_IDX = 1  # Anchor index (defined by order in model attribute)
    ROW = 2  # Row index in grid
    COL = 3  # Column index in grid
    W = 4
    H = 5


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    return inter / (
        area1[:, None] + area2 - inter
    )  # iou = inter / (area1 + area2 - inter)


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def transform_model_outputs_into_predictions(outputs: torch.tensor) -> torch.tensor:
    """Transform model outputs into proper cx, cy, w, h

    For xy we apply a sigmoid and a translation from (0,1) -> (-0.5, 1.5). This means that the
    model can correct each anchor point to be 0.5 positions in the grid smaller or 1.5 positions
    bigger (i.e., move anchor to the middle of the following grid cell, for instance). This is
    also related to what anchors are considered as potential predictions in the loss (in the
    function `find_predicted_boxes`). How we select them are the boxes that the model could modify
    to put the anchor of the grid where the target anchor is.

    For wh, we effectively make the model be able to make anchor boxes from 0 to 4 times bigger
    than the original size. This seems to be related to the filter in size proportion we also
    do on the loss.

    :param outputs: Outputs of a fpn head, last dimension is 5 + num_classes (see PredIdx class).
    :return preds: Tensor of num_preds * (5 + num_classes)
    """
    preds_xy = outputs[..., [PredIdx.CX, PredIdx.CY]].sigmoid() * 2.0 - 0.5
    preds_wh = (outputs[..., [PredIdx.W, PredIdx.H]].sigmoid() * 2) ** 2
    preds_rest = outputs[..., PredIdx.OBJ :]
    preds = torch.cat([preds_xy, preds_wh, preds_rest], dim=-1)
    return preds


class Yolov7Loss:
    MIN_PREDS_FOR_OTA_DYNAMIC_K = 10
    ANCHORS_PER_TARGET = 3

    def __init__(
        self,
        model,
        image_size=ORIG_IMAGE_SIZE,
        box_loss_weight=0.05,
        cls_loss_weight=0.3,
        obj_loss_weight=0.7,
        max_anchor_box_target_size_ratio=4,
    ):
        detection_head = model.detection_head
        self.num_layers = detection_head.num_layers
        self.anchor_sizes_per_layer = detection_head.anchor_sizes_per_layer
        self.stride_per_layer = detection_head.strides
        self.num_anchor_sizes = detection_head.num_anchor_sizes
        self.num_classes = detection_head.num_classes

        # Hardcoded in the original Yolov7 code released with the paper
        if self.num_layers == 3:
            self.obj_loss_layer_weights = [4.0, 1.0, 0.4]
        else:
            self.obj_loss_layer_weights = [4.0, 1.0, 0.25, 0.06, 0.02]

        layer_factor = 3.0 / self.num_layers
        image_size_factor = (image_size / ORIG_IMAGE_SIZE) ** 2
        num_classes_factor = self.num_classes / COCO_NUM_CLASSES
        self.box_loss_weight = box_loss_weight * layer_factor
        self.obj_loss_weight = obj_loss_weight * image_size_factor * layer_factor
        self.cls_loss_weight = cls_loss_weight * num_classes_factor * layer_factor

        self.max_anchor_box_target_size_ratio = max_anchor_box_target_size_ratio

        self.BCEwithLogits = nn.BCEWithLogitsLoss()

        self.training = True
        self.train()

    def __call__(
        self, fpn_heads_outputs, targets, images=None
    ):  # predictions, targets, model
        box_loss, obj_loss, cls_loss = self._compute_losses(
            fpn_heads_outputs, targets, images=images
        )

        batch_size = fpn_heads_outputs[0].shape[0]
        final_loss, loss_items = self._aggregate_losses(
            box_loss, obj_loss, cls_loss, batch_size
        )
        return final_loss, loss_items

    def train(self):
        self.training = True
        self._compute_losses = self._compute_losses_for_train

    def eval(self):
        self.training = False
        self._compute_losses = self._compute_losses_for_eval

    def to(self, device):
        self.BCEwithLogits.to(device)
        # TODO: These are responsibilitity of the model
        self.anchor_sizes_per_layer = self.anchor_sizes_per_layer.to(device)
        self.stride_per_layer = self.stride_per_layer.to(device)

    def _compute_losses_for_train(self, fpn_heads_outputs, targets, images):
        device = targets.device
        box_loss = torch.tensor([0.0], device=device)
        cls_loss = torch.tensor([0.0], device=device)
        obj_loss = torch.tensor([0.0], device=device)

        anchor_boxes_per_layer, _ = self.find_center_prior(
            fpn_heads_outputs, targets, n_anchors_per_target=self.ANCHORS_PER_TARGET
        )
        anchor_boxes_per_layer, targets_per_layer = self.simOTA_assignment(
            fpn_heads_outputs, targets, images, anchor_boxes_per_layer
        )
        for layer_idx, fpn_head_outputs in enumerate(fpn_heads_outputs):
            (
                layer_box_loss,
                layer_obj_loss,
                layer_cls_loss,
            ) = self._compute_fpn_head_losses(
                fpn_head_outputs=fpn_head_outputs,
                anchor_boxes=anchor_boxes_per_layer[layer_idx],
                attempted_targets=targets_per_layer[layer_idx],
                device=device,
            )
            box_loss += layer_box_loss
            cls_loss += layer_cls_loss
            obj_loss += layer_obj_loss * self.obj_loss_layer_weights[layer_idx]
        return box_loss, obj_loss, cls_loss

    def _compute_losses_for_eval(self, fpn_heads_outputs, targets, **_):
        device = targets.device
        box_loss = torch.tensor([0.0], device=device)
        cls_loss = torch.tensor([0.0], device=device)
        obj_loss = torch.tensor([0.0], device=device)

        anchor_boxes_per_layer, targets_per_layer = self.find_center_prior(
            fpn_heads_outputs, targets, n_anchors_per_target=self.ANCHORS_PER_TARGET
        )
        for layer_idx, fpn_head_outputs in enumerate(
            fpn_heads_outputs[: self.num_layers]
        ):
            (
                layer_box_loss,
                layer_obj_loss,
                layer_cls_loss,
            ) = self._compute_fpn_head_losses(
                fpn_head_outputs=fpn_head_outputs,
                anchor_boxes=anchor_boxes_per_layer[layer_idx],
                attempted_targets=targets_per_layer[layer_idx],
                device=device,
            )
            box_loss += layer_box_loss
            cls_loss += layer_cls_loss
            obj_loss += layer_obj_loss * self.obj_loss_layer_weights[layer_idx]
        return box_loss, obj_loss, cls_loss

    def _aggregate_losses(self, box_loss, obj_loss, cls_loss, batch_size):
        box_loss *= self.box_loss_weight
        obj_loss *= self.obj_loss_weight
        cls_loss *= self.cls_loss_weight

        loss = box_loss + obj_loss + cls_loss
        # TODO: Why the batch size scaling?
        final_loss = batch_size * loss
        loss_items = torch.cat([box_loss, obj_loss, cls_loss, loss]).detach()

        return final_loss, loss_items

    def find_center_prior(
        self,
        fpn_heads_outputs: List[torch.tensor],
        targets: torch.tensor,
        n_anchors_per_target: int,
    ) -> Tuple[List]:
        """Identify the anchor boxes for each fpn head to be considered predictions for each target

        Yolov7 is based on anchor boxes, which means that, for each image, there are boxes predicted
        at each anchor of each fpn head grid. Instead of considering all of them for computing
        all losses, a subset is selected in this function. For the subset selected, the regression
        classification and objectness losses will be applied. For the rest, just the objectness.
        For OTA losses, further selection is used (see docs there).

        Each anchor (coordinates in grid) defines a grid cell. Each target, at each fpn head, will
        rarely fall exactly at the anchor. It will fall somewhere inside the cell. Each cell,
        has 4 neighbour cells (top, bottom, left, right). If `n_anchors_per_target=5`, we will
        consider all neighbour anchors. If `n_anchors_per_target=3`, we will consider only the two
        closest to where the target coordinates fall (left or right, and top or bottom).

        Each fpn head has a set of anchor boxes. For each target, we will only consider the subset
        of these anchor boxes that are not much bigger or much smaller. How much is defined by
        the hyperparameter `anchor_t` and the comparison is applied side-wise (not area).

        Finally, the predicted anchor boxes that are considered a potential prediction for the loss
        are:
            - anchor boxes in size for target
            - on all the selected anchors
            - for each fpn head
            - for each target in image
            - for each image in batch

        This is probably center prior.

        :param fpn_heads_outputs: List with outputs for each fpn head.
            Outputs have dims: [n_images, n_anchor_boxes, n_grid_rows, n_grid_cols, n_features]
        :param targets: Target boxes in images in batch. Dims [n_targets, target_vector]
        :param n_anchors_per_target: 3 or 5. Anchors to consider for each target.
        :returns: Tuple with lists, each one with one element per fpn head.
            - target_class_ids_per_layer: class ids of target corresponding to each selected box.
            - target_boxes_per_layer: box [cx, cy, h, w]
                - cx, cy between [0,1] where target box corresponds to falls in the grid cell.
                - h, w in grid space of the target box.
                - Each target is duplicated for all the boxes selected for it.
            - indices_per_layer: Tuple with indices that we will need down the line:
                - image_idxs: Index of the image in batch where the box belongs to.
                - used_anchor_box_idxs: Index of anchor box that corresponds to each selected box.
                    The index comes from the position in self.anchor_sizes_per_layer[fpn_idx] the box comes from.
                - ta_grid_xtr_j: Grid row index for the selected anchor box.
                - ta_grid_xtr_i: Grid column index for the selected anchor box.
            - used_anchor_boxes_per_layer: w,h of the anchor box.

        """
        if n_anchors_per_target == 3:
            anchor_offset = 0.5
        elif n_anchors_per_target == 5:
            anchor_offset = 1.0
        else:
            raise ValueError(
                f"Only 3 or 5 anchor boxes per target are supported: {n_anchors_per_target}"
            )
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        num_targets = targets.shape[0]
        targets_per_layer = []
        anchor_boxes_per_layer = []

        # One row per anchor, one column per target each, 0, 1, 2 in each row
        anchor_box_idxs = (
            torch.arange(self.num_anchor_sizes, device=targets.device)
            .float()
            .repeat_interleave(num_targets)
            .view(self.num_anchor_sizes, num_targets, 1)
        )
        # Duplicate targets for all anchor boxes and add anchor_box_idx as last target component
        target_anchor_box_pairs = torch.cat(
            [targets.repeat(self.num_anchor_sizes, 1, 1), anchor_box_idxs], dim=2
        )

        anchor_offsets = (
            anchor_offset
            * torch.tensor(
                [
                    [0, 0],  # No offset
                    [1, 0],  # x offset for getting left anchor
                    [0, 1],  # y offset for getting top anchor
                    [-1, 0],  # x offset for getting right anchor
                    [0, -1],  # y offset for getting bottom anchor
                ],
                device=targets.device,
            ).float()
        )

        for layer_idx in range(self.num_layers):
            layer_anchor_box_sizes = self.anchor_sizes_per_layer[layer_idx]
            # Note on matrix, rows are y, cols are x (hence 3,2 are inverted)
            grid_size = torch.tensor(fpn_heads_outputs[layer_idx].shape)[[3, 2]]

            # target-anchor pairs on grid coordinates (instead of normalized)
            ta_grid = target_anchor_box_pairs.clone()
            ta_grid[:, :, [TargetIdx.CX, TargetIdx.W]] *= grid_size[0]
            ta_grid[:, :, [TargetIdx.CY, TargetIdx.H]] *= grid_size[1]

            if num_targets > 0:
                size_ratio = (
                    ta_grid[:, :, [TargetIdx.W, TargetIdx.H]]
                    / layer_anchor_box_sizes[:, None, :]
                )
                symmetric_size_ratio = torch.max(size_ratio, 1.0 / size_ratio)
                in_desired_ratio = (
                    symmetric_size_ratio.max(dim=2)[0]
                    < self.max_anchor_box_target_size_ratio
                )
                # Drop anchor_box-target pairs where ratio in either h or w is above the hyperparam
                ta_grid = ta_grid[in_desired_ratio]

                ta_grid_cx = ta_grid[:, TargetIdx.CX]
                ta_grid_cx_inv = grid_size[0] - ta_grid_cx
                ta_grid_cy = ta_grid[:, TargetIdx.CY]
                ta_grid_cy_inv = grid_size[1] - ta_grid_cy

                # Based on where the target coordinates end inside the grid cell, we define the
                #   extra anchor points we will consider for the target (on top of the grid cell).
                #   - anchor_offset=0.5 -> Get closest grid cells (right or left, and top or down)
                #   - anchor_offset=1.0 -> Get all surrounding grid cells (right, left, top, down)
                get_left_anchor = ((ta_grid_cx % 1.0) < anchor_offset) & (
                    ta_grid_cx > 1.0
                )
                get_left_anchor.permute(*torch.arange(get_left_anchor.ndim - 1, -1, -1))
                get_up_anchor = ((ta_grid_cy % 1.0) < anchor_offset) & (
                    ta_grid_cy > 1.0
                )
                get_up_anchor.permute(*torch.arange(get_up_anchor.ndim - 1, -1, -1))
                get_right_anchor = ((ta_grid_cx_inv % 1.0) < anchor_offset) & (
                    ta_grid_cx_inv > 1.0
                )
                get_right_anchor.permute(
                    *torch.arange(get_right_anchor.ndim - 1, -1, -1)
                )
                get_down_anchor = ((ta_grid_cy_inv % 1.0) < anchor_offset) & (
                    ta_grid_cy_inv > 1.0
                )
                get_down_anchor.permute(*torch.arange(get_down_anchor.ndim - 1, -1, -1))

                # Center anchor is the anchor of the grid cell where the target coordinates fall in
                get_center_anchor = torch.ones_like(
                    get_left_anchor
                )  # True for all targets
                extra_anchors_selector = torch.stack(
                    [
                        get_center_anchor,
                        get_left_anchor,
                        get_up_anchor,
                        get_right_anchor,
                        get_down_anchor,
                    ]
                )

                # Duplicate target-anchor_box pair for all possible extra anchor points
                ta_grid_xtr = ta_grid.repeat((extra_anchors_selector.shape[0], 1, 1))
                # Keep only the extra anchors that fulfilled the condition in the selector.
                ta_grid_xtr = ta_grid_xtr[extra_anchors_selector]

                # Duplicate the offsets for each possible target-anchor pair (center and each side)
                offsets_per_ta_pair = anchor_offsets[:, None, :].repeat(
                    1, ta_grid.shape[0], 1
                )
                # Keep only the ones are relevant
                offsets_per_ta_pair = offsets_per_ta_pair[extra_anchors_selector]
            else:
                ta_grid_xtr = target_anchor_box_pairs[0]
                offsets_per_ta_pair = 0

            # Define
            image_idxs, target_class_ids = (
                ta_grid_xtr[:, [TargetIdx.IMG_IDX, TargetIdx.CLS_ID]].long().T
            )
            ta_grid_xtr_xy = ta_grid_xtr[:, [TargetIdx.CX, TargetIdx.CY]]
            ta_grid_xtr_wh = ta_grid_xtr[:, [TargetIdx.W, TargetIdx.H]]

            # Apply the offsets and get the anchor indices (i & j) that each anchor represents in grid
            ta_grid_xtr_ij = (ta_grid_xtr_xy - offsets_per_ta_pair).long()
            ta_grid_xtr_i, ta_grid_xtr_j = ta_grid_xtr_ij.T
            # Ensure we can use them as indexers by clamping
            # TODO: [INPLACE] Original code uses inplace clamp, verify if it makes a difference
            ta_grid_xtr_i.clamp_(0, grid_size[0] - 1)
            ta_grid_xtr_j.clamp_(0, grid_size[1] - 1)

            # Extract the selected boxes, where xy are the position inside the grid cell
            layer_target_boxes = torch.cat(
                (ta_grid_xtr_xy - ta_grid_xtr_ij, ta_grid_xtr_wh), 1
            )
            # What anchor box was used for each one of the predicted boxes
            used_anchor_box_idxs = ta_grid_xtr[:, -1].long()
            used_anchor_boxes = layer_anchor_box_sizes[used_anchor_box_idxs]

            layer_anchor_boxes = torch.column_stack(
                [
                    image_idxs,
                    used_anchor_box_idxs,
                    ta_grid_xtr_i,
                    ta_grid_xtr_j,
                    used_anchor_boxes[:, 0],
                    used_anchor_boxes[:, 1],
                ],
            )

            layer_targets = torch.column_stack(
                [
                    image_idxs,
                    target_class_ids,
                    layer_target_boxes[:, 0],
                    layer_target_boxes[:, 1],
                    layer_target_boxes[:, 2],
                    layer_target_boxes[:, 3],
                ]
            )

            anchor_boxes_per_layer.append(layer_anchor_boxes)
            targets_per_layer.append(layer_targets)
        return anchor_boxes_per_layer, targets_per_layer

    def simOTA_assignment(
        self, fpn_heads_outputs, targets, images, anchor_boxes_per_layer
    ):
        """Start with selected boxes as regular loss, but then, for each image, treat it as an OTA
        problem and select less boxes based on cost. Return those."""
        num_fpn_heads = len(fpn_heads_outputs)
        matched_anchor_boxes_per_layer = [[] for _ in range(num_fpn_heads)]
        matched_targets_per_layer = [[] for _ in range(num_fpn_heads)]

        num_images_in_batch = fpn_heads_outputs[0].shape[0]
        for image_idx in range(num_images_in_batch):

            is_this_image = targets[:, TargetIdx.IMG_IDX] == image_idx
            image_targets = targets[is_this_image]
            image_size = images[image_idx].shape[1]

            num_image_targets = image_targets.shape[0]
            if num_image_targets == 0:
                continue

            image_preds_per_layer = []
            image_anchor_boxes_per_layer = []
            layer_idxs_per_layer = []

            # For each fpn head
            for layer_idx, fpn_head_outputs in enumerate(fpn_heads_outputs):

                is_this_image = (
                    anchor_boxes_per_layer[layer_idx][:, AnchorIdx.IMG_IDX] == image_idx
                )
                image_anchor_boxes = anchor_boxes_per_layer[layer_idx][is_this_image, :]
                image_anchor_boxes_per_layer.append(image_anchor_boxes)

                output_selector = (
                    image_anchor_boxes[
                        :,
                        [
                            AnchorIdx.IMG_IDX,
                            AnchorIdx.ANCHOR_IDX,
                            AnchorIdx.COL,
                            AnchorIdx.ROW,
                        ],
                    ]
                    .long()
                    .T.tolist()
                )
                selected_outputs = fpn_head_outputs[output_selector]

                image_preds = transform_model_outputs_into_predictions(selected_outputs)
                image_preds_per_layer.append(image_preds)

                layer_idxs = torch.full(
                    [image_anchor_boxes.shape[0]], fill_value=layer_idx
                )
                layer_idxs_per_layer.append(layer_idxs)

            layer_idxs = torch.cat(layer_idxs_per_layer, dim=0)
            image_anchor_boxes = torch.cat(image_anchor_boxes_per_layer, dim=0)
            image_preds = torch.cat(image_preds_per_layer, dim=0)
            num_image_preds = image_preds.shape[0]
            if num_image_preds == 0:
                continue

            image_targets_xywh = image_targets[
                :, [TargetIdx.CX, TargetIdx.CY, TargetIdx.W, TargetIdx.H]
            ]
            image_targets_xywh_img_coords = image_targets_xywh * image_size
            image_targets_xyxy = torchvision.ops.box_convert(
                image_targets_xywh_img_coords, "xywh", "xyxy"
            )

            image_preds[..., PredIdx.CX] += image_anchor_boxes[..., AnchorIdx.ROW]
            image_preds[..., PredIdx.CY] += image_anchor_boxes[..., AnchorIdx.COL]
            image_preds[..., [PredIdx.W, PredIdx.H]] *= image_anchor_boxes[
                ..., [AnchorIdx.W, AnchorIdx.H]
            ]
            image_preds_xywh = image_preds[
                ..., [PredIdx.CX, PredIdx.CY, PredIdx.W, PredIdx.H]
            ]
            image_preds_xywh_img_coords = (
                image_preds_xywh * self.stride_per_layer[layer_idxs][:, None]
            )
            image_preds_xyxy = torchvision.ops.box_convert(
                image_preds_xywh_img_coords, "xywh", "xyxy"
            )

            pair_wise_iou = box_iou(image_targets_xyxy, image_preds_xyxy)
            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            # Calculate dynamic K per target as defined in the OTA paper
            k = min(self.MIN_PREDS_FOR_OTA_DYNAMIC_K, pair_wise_iou.shape[1])
            top_k_ious, _ = torch.topk(pair_wise_iou, k, dim=1)
            dynamic_ks = torch.clamp(top_k_ious.sum(dim=1).int(), min=1)

            target_class_probs = (
                F.one_hot(image_targets[:, TargetIdx.CLS_ID].long(), self.num_classes)
                .float()
                .unsqueeze(dim=1)
                .repeat(1, num_image_preds, 1)
            )
            # TODO: Find why they do geometric mean (and test if inplace is needed)
            pred_class_scores = image_preds[:, PredIdx.OBJ + 1 :]
            pred_objectness = image_preds[:, [PredIdx.OBJ]]
            pred_class_probs = (
                (pred_class_scores.sigmoid_() * pred_objectness.sigmoid_())
                .sqrt_()
                .unsqueeze(dim=0)
                .repeat(num_image_targets, 1, 1)
            )

            pair_wise_cls_loss = F.binary_cross_entropy(
                pred_class_probs, target_class_probs, reduction="none"
            ).sum(-1)

            # TODO: This 3.0 is probably a constant that should be named
            cost = (
                pair_wise_cls_loss + 3.0 * pair_wise_iou_loss
            )  # num_targets x num_preds

            match_matrix = torch.zeros_like(cost).long()

            for target_idx in range(num_image_targets):
                # For each gt, assign k boxes with lowest cost. K comes from sum of IoU above.
                target_k = dynamic_ks[target_idx].item()
                _, pred_idxs = torch.topk(cost[target_idx], k=target_k, largest=False)
                match_matrix[target_idx][pred_idxs] = 1

            # del top_k, dynamic_ks
            num_targets_per_pred = match_matrix.sum(dim=0)
            is_ambigous_pred = num_targets_per_pred > 1
            if is_ambigous_pred.sum() > 0:
                _, min_cost_target_idxs = torch.min(cost[:, is_ambigous_pred], dim=0)
                match_matrix[:, is_ambigous_pred] = 0
                match_matrix[min_cost_target_idxs, is_ambigous_pred] = 1
            pred_is_matched = match_matrix.sum(dim=0) > 0
            pred_matched_target_idxs = match_matrix[:, pred_is_matched].argmax(dim=0)

            # Keep only prediction boxes selected by the dynamic ks
            layer_idxs = layer_idxs[pred_is_matched.to(device=layer_idxs.device)]
            image_matched_anchor_boxes = image_anchor_boxes[pred_is_matched]

            # Duplicate targets as many times as boxes assigned, tensor has size of finally assigned boxes
            image_matched_targets = image_targets[pred_matched_target_idxs]

            for layer_idx in range(num_fpn_heads):
                is_this_layer = layer_idxs == layer_idx
                matched_targets_per_layer[layer_idx].append(
                    image_matched_targets[is_this_layer]
                )
                matched_anchor_boxes_per_layer[layer_idx].append(
                    image_matched_anchor_boxes[is_this_layer]
                )

        # At the end, not in image by image loop
        for layer_idx in range(num_fpn_heads):
            # If any box assigned from the layer
            if matched_targets_per_layer[layer_idx]:
                # Note on matrix, rows are y, cols are x (hence 3,2 are inverted)
                grid_size = torch.tensor(fpn_heads_outputs[layer_idx].shape)[[3, 2]]
                layer_targets = torch.cat(matched_targets_per_layer[layer_idx], dim=0)
                layer_anchor_boxes = torch.cat(
                    matched_anchor_boxes_per_layer[layer_idx], dim=0
                )

                layer_targets[:, [TargetIdx.CX, TargetIdx.W]] *= grid_size[0]
                layer_targets[:, [TargetIdx.CY, TargetIdx.H]] *= grid_size[1]
                layer_targets[:, [TargetIdx.CX, TargetIdx.CY]] -= layer_anchor_boxes[
                    :, [AnchorIdx.ROW, AnchorIdx.COL]
                ]

                matched_targets_per_layer[layer_idx] = layer_targets
                matched_anchor_boxes_per_layer[layer_idx] = layer_anchor_boxes
            else:
                matched_targets_per_layer[layer_idx] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )
                matched_anchor_boxes_per_layer[layer_idx] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )
        return matched_anchor_boxes_per_layer, matched_targets_per_layer

    def _compute_fpn_head_losses(
        self, fpn_head_outputs, anchor_boxes, attempted_targets, device
    ):
        # Initialize in case the conditions for them are not hit
        box_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)

        target_objectness = torch.zeros_like(fpn_head_outputs[..., 0], device=device)

        num_anchor_boxes = anchor_boxes.shape[0]
        if num_anchor_boxes > 0:
            output_selector = (
                anchor_boxes[
                    :,
                    [
                        AnchorIdx.IMG_IDX,
                        AnchorIdx.ANCHOR_IDX,
                        AnchorIdx.COL,
                        AnchorIdx.ROW,
                    ],
                ]
                .long()
                .T.tolist()
            )
            selected_outputs = fpn_head_outputs[output_selector]

            preds = transform_model_outputs_into_predictions(selected_outputs)
            preds[..., [PredIdx.W, PredIdx.H]] *= anchor_boxes[
                :, [AnchorIdx.W, AnchorIdx.H]
            ]
            pred_boxes = preds[..., [PredIdx.CX, PredIdx.CY, PredIdx.W, PredIdx.H]]
            target_boxes = attempted_targets[
                ..., [TargetIdx.CX, TargetIdx.CY, TargetIdx.W, TargetIdx.H]
            ]
            # TODO: Transpose is because fun transposes 1 (??), fix when function cleaned
            iou = bbox_iou(pred_boxes.T, target_boxes, x1y1x2y2=False, CIoU=True)

            box_loss = (1.0 - iou).mean()

            # Objectness
            # TODO: Why detach? No backprop from IoU? Consider it a constant?
            target_objectness[output_selector] = (
                iou.detach().clamp(0).type(target_objectness.dtype)
            )
            # Classification
            if self.num_classes > 1:  # cls loss (only if multiple classes)
                pred_class_probs = preds[..., PredIdx.OBJ + 1 :]
                # One-hot encode the target class
                target_class_probs = torch.full_like(pred_class_probs, 0, device=device)
                target_class_ids = attempted_targets[:, TargetIdx.CLS_ID].long()
                target_class_probs[range(num_anchor_boxes), target_class_ids] = 1
                cls_loss = self.BCEwithLogits(pred_class_probs, target_class_probs)

        pred_objectness = fpn_head_outputs[..., PredIdx.OBJ]
        obj_loss = self.BCEwithLogits(pred_objectness, target_objectness)

        return box_loss, obj_loss, cls_loss


class Yolov7LossWithAux(Yolov7Loss):

    AUX_WEIGHT = 0.25
    MIN_PREDS_FOR_OTA_DYNAMIC_K = 20
    ANCHORS_PER_TARGET = 3
    AUX_ANCHORS_PER_TARGET = 5

    def _compute_losses_for_train(self, fpn_heads_outputs, targets, images):
        device = targets.device
        box_loss = torch.tensor([0.0], device=device)
        cls_loss = torch.tensor([0.0], device=device)
        obj_loss = torch.tensor([0.0], device=device)

        anchor_boxes_per_layer, _ = self.find_center_prior(
            fpn_heads_outputs[: self.num_layers],
            targets,
            n_anchors_per_target=self.ANCHORS_PER_TARGET,
        )
        anchor_boxes_per_layer, targets_per_layer = self.simOTA_assignment(
            fpn_heads_outputs[: self.num_layers],
            targets,
            images,
            anchor_boxes_per_layer,
        )
        aux_anchor_boxes_per_layer, _ = self.find_center_prior(
            fpn_heads_outputs[: self.num_layers],
            targets,
            n_anchors_per_target=self.AUX_ANCHORS_PER_TARGET,
        )
        aux_anchor_boxes_per_layer, aux_targets_per_layer = self.simOTA_assignment(
            fpn_heads_outputs[: self.num_layers],
            targets,
            images,
            aux_anchor_boxes_per_layer,
        )
        for layer_idx, fpn_head_outputs in enumerate(
            fpn_heads_outputs[: self.num_layers]
        ):
            (
                layer_box_loss,
                layer_obj_loss,
                layer_cls_loss,
            ) = self._compute_fpn_head_losses(
                fpn_head_outputs=fpn_head_outputs,
                anchor_boxes=anchor_boxes_per_layer[layer_idx],
                attempted_targets=targets_per_layer[layer_idx],
                device=device,
            )

            aux_fpn_head_outputs = fpn_heads_outputs[layer_idx + self.num_layers]
            (
                aux_layer_box_loss,
                aux_layer_obj_loss,
                aux_layer_cls_loss,
            ) = self._compute_fpn_head_losses(
                fpn_head_outputs=aux_fpn_head_outputs,
                anchor_boxes=aux_anchor_boxes_per_layer[layer_idx],
                attempted_targets=aux_targets_per_layer[layer_idx],
                device=device,
            )

            box_loss += layer_box_loss + self.AUX_WEIGHT * aux_layer_box_loss
            cls_loss += layer_cls_loss + self.AUX_WEIGHT * aux_layer_cls_loss
            obj_loss += (
                layer_obj_loss + self.AUX_WEIGHT * aux_layer_obj_loss
            ) * self.obj_loss_layer_weights[layer_idx]
        return box_loss, obj_loss, cls_loss


# TODO: To method to reproduce trainer behavior start run.
