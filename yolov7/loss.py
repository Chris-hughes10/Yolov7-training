# copied from https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py
# Loss functions
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.ops.boxes import box_iou

from yolov7.migrated.utils.general import (
    xywh2xyxy,
)
from yolov7.migrated.utils.torch_utils import is_parallel

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


def smooth_BCE(
    eps=0.1,
):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        # I believe p_t is actually equal to loss (unless pytorch applies log to p_t, which according to docs doesnt look like)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

def transform_model_outputs_into_predictions(outputs: torch.tensor) -> torch.tensor:
    """Transform moraw outputs into proper cx, cy, w, h

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
    preds_rest = outputs[..., PredIdx.OBJ:]
    preds = torch.cat([preds_xy, preds_wh, preds_rest], dim=1)
    return preds


class ComputeYolov7Loss:
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        self.BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["cls_pw"]], device=device)
        )
        self.BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["obj_pw"]], device=device)
        )

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # cp -> Value used as prob for positive (1.0)
        # cn -> Value used as prob for negative (0.0)
        self.cp, self.cn = smooth_BCE(
            eps=h.get("label_smoothing", 0.0)
        )  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            self.BCEcls = FocalLoss(self.BCEcls, g)
            self.BCEobj = FocalLoss(self.BCEobj, g)

        det = (
            model.module.model[-1] if is_parallel(model) else model.model[-1]
        )  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            det.nl, [4.0, 1.0, 0.25, 0.06, 0.02]
        )  # P3-P7

        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.gr = model.gr
        self.hyp = h
        self.autobalance = autobalance
        self.na = det.na
        self.nc = det.nc
        self.nl = det.nl
        self.anchors = det.anchors

    def old_compute_losses(self, p, targets, lcls, lbox, lobj, device, **kwargs):
        # tcls, tbox, indices, anchors = self.find_n_positive(p, targets)  # targets
        tcls, tbox, indices, anchors = self.find_predicted_boxes(p, targets)

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(
                    pbox.T, tbox[i], x1y1x2y2=False, CIoU=True
                )  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(
                    0
                ).type(
                    tobj.dtype
                )  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = (
                    self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                )

        return tobj

    def prev_compute_losses(self, fpn_heads_outputs, targets, lcls, lbox, lobj, device, **kwargs):
        # tcls, tbox, indices, anchors = self.find_n_positive(p, targets)  # targets
        found_boxes = self.find_predicted_boxes(fpn_heads_outputs, targets)
        target_class_ids_per_layer = found_boxes[0]
        target_boxes_per_layer = found_boxes[1]
        indices_per_layer = found_boxes[2]
        used_anchor_boxes_per_layer = found_boxes[3]

        # Losses
        for layer_idx, fpn_head_outputs in enumerate(fpn_heads_outputs):
            image_idxs, anchor_box_idxs, grid_j, grid_i = indices_per_layer[layer_idx]
            target_objectness = torch.zeros_like(fpn_head_outputs[..., 0], device=device)

            num_anchor_boxes = image_idxs.shape[0]
            if num_anchor_boxes > 0:
                selected_outputs = fpn_head_outputs[image_idxs, anchor_box_idxs, grid_j, grid_i]

                preds = transform_model_outputs_into_predictions(selected_outputs)
                preds[..., [PredIdx.W, PredIdx.H]] *= used_anchor_boxes_per_layer[layer_idx]
                pred_boxes = preds[..., [PredIdx.CX, PredIdx.CY, PredIdx.W, PredIdx.H]]
                target_boxes = target_boxes_per_layer[layer_idx]
                # TODO: Transpose is because fun transposes 1 (??), fix when function cleaned
                iou = bbox_iou(pred_boxes.T, target_boxes, x1y1x2y2=False, CIoU=True)

                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                # TODO: Why detach? No backprop from IoU? Consider it a constant?
                target_objectness[image_idxs, anchor_box_idxs, grid_j, grid_i] = (
                    (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(target_objectness.dtype)
                )
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    pred_class_probs = preds[..., PredIdx.OBJ+1:]
                    # TODO: SmoothBCE might not be used, drop if neccessary
                    target_class_probs = torch.full_like(pred_class_probs, self.cn, device=device)
                    target_class_ids = target_class_ids_per_layer[layer_idx]
                    target_class_probs[range(num_anchor_boxes), target_class_ids] = self.cp
                    lcls += self.BCEcls(pred_class_probs, target_class_probs)

            pred_objectness = fpn_head_outputs[..., PredIdx.OBJ]
            layer_objectness_loss = self.BCEobj(pred_objectness, target_objectness)
            lobj += layer_objectness_loss * self.balance[layer_idx]  # obj loss
            if self.autobalance:
                self.balance[layer_idx] = (
                        self.balance[layer_idx] * 0.9999 + 0.0001 / layer_objectness_loss.detach().item()
                )

        return target_objectness

    def compute_losses(self, fpn_heads_outputs, targets, lcls, lbox, lobj, device, **kwargs):
        # tcls, tbox, indices, anchors = self.find_n_positive(p, targets)  # targets
        anchor_boxes_per_layer, targets_per_layer = self.find_predicted_boxes(fpn_heads_outputs, targets, **kwargs)

        # Losses
        for layer_idx, fpn_head_outputs in enumerate(fpn_heads_outputs):
            anchor_boxes = anchor_boxes_per_layer[layer_idx]
            attempted_targets = targets_per_layer[layer_idx]
            target_objectness = torch.zeros_like(fpn_head_outputs[..., 0], device=device)

            num_anchor_boxes = anchor_boxes.shape[0]
            if num_anchor_boxes > 0:
                output_selector = anchor_boxes[:, [AnchorIdx.IMG_IDX, AnchorIdx.ANCHOR_IDX, AnchorIdx.COL, AnchorIdx.ROW]].long().T.tolist()
                selected_outputs = fpn_head_outputs[output_selector]

                preds = transform_model_outputs_into_predictions(selected_outputs)
                preds[..., [PredIdx.W, PredIdx.H]] *= anchor_boxes[:, [AnchorIdx.W, AnchorIdx.H]]
                pred_boxes = preds[..., [PredIdx.CX, PredIdx.CY, PredIdx.W, PredIdx.H]]
                target_boxes = attempted_targets[..., [TargetIdx.CX, TargetIdx.CY, TargetIdx.W, TargetIdx.H]]
                # TODO: Transpose is because fun transposes 1 (??), fix when function cleaned
                iou = bbox_iou(pred_boxes.T, target_boxes, x1y1x2y2=False, CIoU=True)

                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                # TODO: Why detach? No backprop from IoU? Consider it a constant?
                target_objectness[output_selector] = (
                    (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(target_objectness.dtype)
                )
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    pred_class_probs = preds[..., PredIdx.OBJ+1:]
                    # TODO: SmoothBCE might not be used, drop if neccessary
                    target_class_probs = torch.full_like(pred_class_probs, self.cn, device=device)
                    target_class_ids = attempted_targets[:, TargetIdx.CLS_ID].long()
                    target_class_probs[range(num_anchor_boxes), target_class_ids] = self.cp
                    lcls += self.BCEcls(pred_class_probs, target_class_probs)

            pred_objectness = fpn_head_outputs[..., PredIdx.OBJ]
            layer_objectness_loss = self.BCEobj(pred_objectness, target_objectness)
            lobj += layer_objectness_loss * self.balance[layer_idx]  # obj loss
            if self.autobalance:
                self.balance[layer_idx] = (
                        self.balance[layer_idx] * 0.9999 + 0.0001 / layer_objectness_loss.detach().item()
                )

        return target_objectness

    def __call__(self, p, targets, **kwargs):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = (
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
        )
        #####

        tobj = self.compute_losses(fpn_heads_outputs=p, targets=targets, lcls=lcls, lbox=lbox,
                                   lobj=lobj, device=device, **kwargs)

        ###
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


    def prev_find_predicted_boxes(
        self,
        fpn_heads_outputs: List[torch.tensor],
        targets: torch.tensor,
        n_anchors_per_target: int=3
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
                    The index comes from the position in self.anchors[fpn_idx] the box comes from.
                - ta_grid_xtr_j: Grid row index for the selected anchor box.
                - ta_grid_xtr_i: Grid column index for the selected anchor box.
            - used_anchor_boxes_per_layer: w,h of the anchor box.

        """
        if n_anchors_per_target == 3:
            anchor_offset = 0.5
        elif n_anchors_per_target == 5:
            anchor_offset = 1.0
        else:
            raise ValueError(f"Only 3 or 5 anchor boxes per target are supported: {n_anchors_per_target}")
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        num_anchor_boxes, num_targets = self.na, targets.shape[0]
        target_boxes_per_layer = []
        target_class_ids_per_layer = []
        indices_per_layer = []
        used_anchor_boxes_per_layer = []

        # One row per anchor, one column per target each, 0, 1, 2 in each row
        anchor_box_idxs = (
            torch.arange(num_anchor_boxes, device=targets.device).float()
            .repeat_interleave(num_targets)
            .view(num_anchor_boxes, num_targets, 1)
        )
        # Duplicate targets for all anchor boxes and add anchor_box_idx as last target component
        target_anchor_box_pairs = torch.cat(
            [targets.repeat(num_anchor_boxes, 1, 1), anchor_box_idxs],
            dim=2
        )

        anchor_offsets = anchor_offset * torch.tensor(
                [
                    [0, 0], # No offset
                    [1, 0], # x offset for getting left anchor
                    [0, 1], # y offset for getting top anchor
                    [-1, 0], # x offset for getting right anchor
                    [0, -1], # y offset for getting bottom anchor
                ],
                device=targets.device,
            ).float()

        for layer_idx in range(self.nl):
            layer_anchor_boxes = self.anchors[layer_idx]
            # Note on matrix, rows are y, cols are x (hence 3,2 are inverted)
            grid_size = torch.tensor(fpn_heads_outputs[layer_idx].shape)[[3, 2]]

            # target-anchor pairs on grid coordinates (instead of normalized)
            ta_grid = target_anchor_box_pairs.clone()
            ta_grid[:, :, [TargetIdx.CX,TargetIdx.W]] *= grid_size[0]
            ta_grid[:, :, [TargetIdx.CY,TargetIdx.H]] *= grid_size[1]

            if num_targets > 0:
                size_ratio = ta_grid[:, :, [TargetIdx.W, TargetIdx.H]] / layer_anchor_boxes[:, None, :]
                symmetric_size_ratio = torch.max(size_ratio, 1.0 / size_ratio)
                in_desired_ratio = symmetric_size_ratio.max(dim=2)[0] < self.hyp["anchor_t"]
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
                get_left_anchor = (((ta_grid_cx % 1.0) < anchor_offset) & (ta_grid_cx > 1.0)).T
                get_up_anchor = (((ta_grid_cy % 1.0) < anchor_offset) & (ta_grid_cy > 1.0)).T
                get_right_anchor = (((ta_grid_cx_inv % 1.0) < anchor_offset) & (ta_grid_cx_inv > 1.0)).T
                get_down_anchor = (((ta_grid_cy_inv % 1.0) < anchor_offset) & (ta_grid_cy_inv > 1.0)).T

                # Center anchor is the anchor of the grid cell where the target coordinates fall in
                get_center_anchor = torch.ones_like(get_left_anchor)  # True for all targets
                extra_anchors_selector = torch.stack(
                    [
                        get_center_anchor,
                        get_left_anchor,
                        get_up_anchor,
                        get_right_anchor,
                        get_down_anchor
                    ]
                )

                # Duplicate target-anchor_box pair for all possible extra anchor points
                ta_grid_xtr = ta_grid.repeat((extra_anchors_selector.shape[0], 1, 1))
                # Keep only the extra anchors that fulfilled the condition in the selector.
                ta_grid_xtr = ta_grid_xtr[extra_anchors_selector]

                # Duplicate the offsets for each possible target-anchor pair (center and each side)
                offsets_per_ta_pair = anchor_offsets[:, None, :].repeat(1, ta_grid.shape[0], 1)
                # Keep only the ones are relevant
                offsets_per_ta_pair = offsets_per_ta_pair[extra_anchors_selector]
            else:
                ta_grid_xtr = targets[0]
                offsets_per_ta_pair = 0

            # Define
            image_idxs, target_class_ids = ta_grid_xtr[:, [TargetIdx.IMG_IDX, TargetIdx.CLS_ID]].long().T
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
            layer_target_boxes = torch.cat((ta_grid_xtr_xy - ta_grid_xtr_ij, ta_grid_xtr_wh), 1)
            # What anchor box was used for each one of the predicted boxes
            used_anchor_box_idxs = ta_grid_xtr[:, -1].long()
            used_anchor_boxes = layer_anchor_boxes[used_anchor_box_idxs]

            # Indices tuple with all indices we will need down the line
            indices = (image_idxs, used_anchor_box_idxs, ta_grid_xtr_j, ta_grid_xtr_i)

            indices_per_layer.append(indices)
            target_boxes_per_layer.append(layer_target_boxes)
            target_class_ids_per_layer.append(target_class_ids)
            used_anchor_boxes_per_layer.append(used_anchor_boxes)

        return (
            target_class_ids_per_layer,
            target_boxes_per_layer,
            indices_per_layer,
            used_anchor_boxes_per_layer
        )

    def find_predicted_boxes(self, fpn_heads_outputs, targets, **kwargs):
        return self.find_center_prior(fpn_heads_outputs, targets)

    def find_center_prior(
        self,
        fpn_heads_outputs: List[torch.tensor],
        targets: torch.tensor,
        n_anchors_per_target: int=3
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
                    The index comes from the position in self.anchors[fpn_idx] the box comes from.
                - ta_grid_xtr_j: Grid row index for the selected anchor box.
                - ta_grid_xtr_i: Grid column index for the selected anchor box.
            - used_anchor_boxes_per_layer: w,h of the anchor box.

        """
        if n_anchors_per_target == 3:
            anchor_offset = 0.5
        elif n_anchors_per_target == 5:
            anchor_offset = 1.0
        else:
            raise ValueError(f"Only 3 or 5 anchor boxes per target are supported: {n_anchors_per_target}")
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        num_anchor_boxes, num_targets = self.na, targets.shape[0]
        targets_per_layer = []
        anchor_boxes_per_layer = []

        # One row per anchor, one column per target each, 0, 1, 2 in each row
        anchor_box_idxs = (
            torch.arange(num_anchor_boxes, device=targets.device).float()
            .repeat_interleave(num_targets)
            .view(num_anchor_boxes, num_targets, 1)
        )
        # Duplicate targets for all anchor boxes and add anchor_box_idx as last target component
        target_anchor_box_pairs = torch.cat(
            [targets.repeat(num_anchor_boxes, 1, 1), anchor_box_idxs],
            dim=2
        )

        anchor_offsets = anchor_offset * torch.tensor(
                [
                    [0, 0], # No offset
                    [1, 0], # x offset for getting left anchor
                    [0, 1], # y offset for getting top anchor
                    [-1, 0], # x offset for getting right anchor
                    [0, -1], # y offset for getting bottom anchor
                ],
                device=targets.device,
            ).float()

        for layer_idx in range(self.nl):
            layer_anchor_box_sizes = self.anchors[layer_idx]
            # Note on matrix, rows are y, cols are x (hence 3,2 are inverted)
            grid_size = torch.tensor(fpn_heads_outputs[layer_idx].shape)[[3, 2]]

            # target-anchor pairs on grid coordinates (instead of normalized)
            ta_grid = target_anchor_box_pairs.clone()
            ta_grid[:, :, [TargetIdx.CX,TargetIdx.W]] *= grid_size[0]
            ta_grid[:, :, [TargetIdx.CY,TargetIdx.H]] *= grid_size[1]

            if num_targets > 0:
                size_ratio = ta_grid[:, :, [TargetIdx.W, TargetIdx.H]] / layer_anchor_box_sizes[:, None, :]
                symmetric_size_ratio = torch.max(size_ratio, 1.0 / size_ratio)
                in_desired_ratio = symmetric_size_ratio.max(dim=2)[0] < self.hyp["anchor_t"]
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
                get_left_anchor = (((ta_grid_cx % 1.0) < anchor_offset) & (ta_grid_cx > 1.0)).T
                get_up_anchor = (((ta_grid_cy % 1.0) < anchor_offset) & (ta_grid_cy > 1.0)).T
                get_right_anchor = (((ta_grid_cx_inv % 1.0) < anchor_offset) & (ta_grid_cx_inv > 1.0)).T
                get_down_anchor = (((ta_grid_cy_inv % 1.0) < anchor_offset) & (ta_grid_cy_inv > 1.0)).T

                # Center anchor is the anchor of the grid cell where the target coordinates fall in
                get_center_anchor = torch.ones_like(get_left_anchor)  # True for all targets
                extra_anchors_selector = torch.stack(
                    [
                        get_center_anchor,
                        get_left_anchor,
                        get_up_anchor,
                        get_right_anchor,
                        get_down_anchor
                    ]
                )

                # Duplicate target-anchor_box pair for all possible extra anchor points
                ta_grid_xtr = ta_grid.repeat((extra_anchors_selector.shape[0], 1, 1))
                # Keep only the extra anchors that fulfilled the condition in the selector.
                ta_grid_xtr = ta_grid_xtr[extra_anchors_selector]

                # Duplicate the offsets for each possible target-anchor pair (center and each side)
                offsets_per_ta_pair = anchor_offsets[:, None, :].repeat(1, ta_grid.shape[0], 1)
                # Keep only the ones are relevant
                offsets_per_ta_pair = offsets_per_ta_pair[extra_anchors_selector]
            else:
                ta_grid_xtr = targets[0]
                offsets_per_ta_pair = 0

            # Define
            image_idxs, target_class_ids = ta_grid_xtr[:, [TargetIdx.IMG_IDX, TargetIdx.CLS_ID]].long().T
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
            layer_target_boxes = torch.cat((ta_grid_xtr_xy - ta_grid_xtr_ij, ta_grid_xtr_wh), 1)
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



class ComputeYolov7LossOTA(ComputeYolov7Loss):
    def __init__(self, model, autobalance=False):
        super().__init__(model, autobalance)
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]
        self.stride = det.stride
        self.min_for_top_k = 10 # Waiting understanding for better naming


    # def __call__(self, p, targets, **kwargs):  # predictions, targets, model
    #     device = targets.device
    #     lcls, lbox, lobj = (
    #         torch.zeros(1, device=device),
    #         torch.zeros(1, device=device),
    #         torch.zeros(1, device=device),
    #     )
    #     #####

    #     tobj = self.compute_losses(p=p, targets=targets, lcls=lcls, lbox=lbox,
    #                                lobj=lobj, device=device, **kwargs)

    #     ###
    #     if self.autobalance:
    #         self.balance = [x / self.balance[self.ssi] for x in self.balance]
    #     lbox *= self.hyp["box"]
    #     lobj *= self.hyp["obj"]
    #     lcls *= self.hyp["cls"]
    #     bs = tobj.shape[0]  # batch size

    #     loss = lbox + lobj + lcls
    #     return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


    def old_compute_losses(self, p, targets, imgs, lcls, lbox, lobj, device, **kwargs):
        """This can probably be unified with the base class if build targets returns built boxes
        with their class, same as find_n_positive."""
        # bs, as_, gjs, gis, targets, anchors = self.prev_build_targets(p, targets, imgs)
        the_anchors, targets = self.build_targets(p, targets, imgs)
        bs = [a[:, AnchorIdx.IMG_IDX].long() for a in the_anchors]
        as_ = [a[:, AnchorIdx.ANCHOR_IDX].long() for a in the_anchors]
        gjs = [a[:, AnchorIdx.COL].long() for a in the_anchors]
        gis = [a[:, AnchorIdx.ROW].long() for a in the_anchors]
        anchors = [a[:, [AnchorIdx.W, AnchorIdx.H]] for a in the_anchors]

        pre_gen_gains = [
            torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p
        ]

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                # pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                grid = torch.stack([gi, gj], dim=1)
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(
                    pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True
                )  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(
                    0
                ).type(
                    tobj.dtype
                )  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        return tobj

    def old_build_targets(self, p, targets, imgs, n_anchor_per_gt=3):

        # _, _, indices, anch = self.find_n_positive(p, targets, n_anchor_per_gt=n_anchor_per_gt)
        _, _, indices, anch = self.find_predicted_boxes(p, targets, n_anchors_per_target=n_anchor_per_gt)

        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]

        nl = len(p)

        for batch_idx in range(p[0].shape[0]):

            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []

            for i, pi in enumerate(p):

                b, a, gj, gi = indices[i]
                idx = b == batch_idx
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)

                fg_pred = pi[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = torch.stack([gi, gj], dim=1)
                # Sigmoid (-inf, inf) -> (0, 1) interval
                #   * 2 -> (0, 2) interval
                #   - 0.5 -> (-0.5, 1.5) interval
                pxy = (fg_pred[:, :2].sigmoid() * 2.0 - 0.5 + grid) * self.stride[
                    i
                ]  # / 8.
                # pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                # (-inf, inf) -> (0, 4) (which coincides with anchor_t param)
                pwh = (
                    (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i]
                )  # / 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(self.min_for_top_k, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_

            cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss

            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )
                matching_as[i] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )
                matching_gjs[i] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )
                matching_gis[i] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )
                matching_targets[i] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )
                matching_anchs[i] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )

        return (
            matching_bs,
            matching_as,
            matching_gjs,
            matching_gis,
            matching_targets,
            matching_anchs,
        )

    def prev_build_targets(self, fpn_heads_outputs, targets, imgs, n_anchor_per_gt=3):
        """Start with selected boxes as regular loss, but then, for each image, treat it as an OTA
        problem and select less boxes based on cost. Return those."""

        # found_boxes = self.find_predicted_boxes(fpn_heads_outputs, targets, n_anchors_per_target=n_anchor_per_gt)
        # indices = found_boxes[2]
        # anchor_boxes_per_layer = found_boxes[3]

        the_boxes, _ = self.find_center_prior(fpn_heads_outputs, targets, n_anchors_per_target=n_anchor_per_gt)


        matching_image_idxs = [[] for fpn_head_outputs in fpn_heads_outputs]
        matching_anchor_idxs = [[] for fpn_head_outputs in fpn_heads_outputs]
        matching_grid_j = [[] for fpn_head_outputs in fpn_heads_outputs]
        matching_grid_i = [[] for fpn_head_outputs in fpn_heads_outputs]
        matching_targets = [[] for fpn_head_outputs in fpn_heads_outputs]
        matching_anchor_boxes = [[] for fpn_head_outputs in fpn_heads_outputs]

        num_fpn_heads = len(fpn_heads_outputs)

        for image_idx in range(fpn_heads_outputs[0].shape[0]):

            is_this_image = targets[:, TargetIdx.IMG_IDX] == image_idx
            image_targets = targets[is_this_image]
            image_size = imgs[image_idx].shape[1]

            num_image_targets = image_targets.shape[0]
            if num_image_targets == 0:
                continue

            targets_xywh = image_targets[:, [TargetIdx.CX, TargetIdx.CY, TargetIdx.W, TargetIdx.H]]
            targets_xywh_img_coords = targets_xywh * image_size
            targets_xyxy = xywh2xyxy(targets_xywh_img_coords)

            preds_xyxy = []

            all_preds_xyxy = []
            all_pred_class_scores = []
            all_pred_objectness = []
            all_layer_idxs = []
            all_image_idxs = []
            all_anchor_box_idxs = []
            all_grid_j = []
            all_grid_i = []
            all_anchor_boxes = []

            # For each fpn head
            for layer_idx, fpn_head_outputs in enumerate(fpn_heads_outputs):
                # image_idxs, anchor_box_idxs, grid_j, grid_i = indices[layer_idx]
                image_idxs = the_boxes[layer_idx][:, AnchorIdx.IMG_IDX].long()
                anchor_box_idxs = the_boxes[layer_idx][:, AnchorIdx.ANCHOR_IDX].long()
                grid_j = the_boxes[layer_idx][:, AnchorIdx.COL].long()
                grid_i = the_boxes[layer_idx][:, AnchorIdx.ROW].long()

                is_this_image = image_idxs == image_idx
                image_idxs = image_idxs[is_this_image]
                anchor_box_idxs = anchor_box_idxs[is_this_image]
                grid_j = grid_j[is_this_image]
                grid_i = grid_i[is_this_image]
                # anchor_boxes = anchor_boxes_per_layer[layer_idx][is_this_image]
                anchor_boxes = the_boxes[layer_idx][:, [AnchorIdx.W, AnchorIdx.H]][is_this_image]
                layer_idxs = torch.full_like(image_idxs, fill_value=layer_idx)

                all_image_idxs.append(image_idxs)
                all_anchor_box_idxs.append(anchor_box_idxs)
                all_grid_j.append(grid_j) # grid j
                all_grid_i.append(grid_i) # grid i
                all_anchor_boxes.append(anchor_boxes) # anchors (w,h)
                all_layer_idxs.append(layer_idxs) # layer_idx

                selected_outputs = fpn_head_outputs[image_idxs, anchor_box_idxs, grid_j, grid_i]

                preds = transform_model_outputs_into_predictions(selected_outputs)
                preds[..., PredIdx.CX] += grid_i
                preds[..., PredIdx.CY] += grid_j
                preds[..., [PredIdx.W, PredIdx.H]] *= anchor_boxes
                preds_xywh = preds[..., [PredIdx.CX, PredIdx.CY, PredIdx.W, PredIdx.H]]
                preds_xywh_img_coords = preds_xywh * self.stride[layer_idx]
                preds_xyxy = xywh2xyxy(preds_xywh_img_coords)
                all_preds_xyxy.append(preds_xyxy)

                pred_objectness = preds[..., [PredIdx.OBJ]]
                all_pred_objectness.append(pred_objectness)

                pred_class_scores = preds[..., PredIdx.OBJ+1:]
                all_pred_class_scores.append(pred_class_scores)

            # All candidate boxes, from all layers, for current image (N x 4, xyxy)
            all_preds_xyxy = torch.cat(all_preds_xyxy, dim=0)
            num_image_preds = all_preds_xyxy.shape[0]
            if num_image_preds == 0:
                continue
            pred_objectness = torch.cat(all_pred_objectness, dim=0)
            pred_class_scores = torch.cat(all_pred_class_scores, dim=0).float()
            layer_idxs = torch.cat(all_layer_idxs, dim=0)
            image_idxs = torch.cat(all_image_idxs, dim=0)
            anchor_box_idxs = torch.cat(all_anchor_box_idxs, dim=0)
            grid_j = torch.cat(all_grid_j, dim=0)
            grid_i = torch.cat(all_grid_i, dim=0)
            anchor_boxes = torch.cat(all_anchor_boxes, dim=0)

            pair_wise_iou = box_iou(targets_xyxy, all_preds_xyxy)
            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            # Calculate dynamic K per target as defined in the OTA paper
            k = min(self.min_for_top_k, pair_wise_iou.shape[1])
            top_k_ious, _ = torch.topk(pair_wise_iou, k, dim=1)
            dynamic_ks = torch.clamp(top_k_ious.sum(dim=1).int(), min=1)

            target_class_scores = (
                F.one_hot(image_targets[:, TargetIdx.CLS_ID].long(), self.nc).float()
                .unsqueeze(dim=1).repeat(1, num_image_preds, 1)
            )
            pred_class_scores = (
                (pred_class_scores.sigmoid_() * pred_objectness.sigmoid_())
                .unsqueeze(dim=0).repeat(num_image_targets, 1, 1)
            )

            # TODO: Find the paper where this is defined
            pred_class_scores = pred_class_scores.sqrt_()
            pred_class_scores = torch.log(pred_class_scores / (1 - pred_class_scores))
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                pred_class_scores, target_class_scores, reduction="none"
            ).sum(-1)
            # del cls_preds_

            # TODO: This 3.0 is probably a constant that should be named
            cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss  # num_targets x num_preds

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
            layer_idxs = layer_idxs[pred_is_matched]
            image_idxs = image_idxs[pred_is_matched]
            anchor_box_idxs = anchor_box_idxs[pred_is_matched]
            grid_j = grid_j[pred_is_matched]
            grid_i = grid_i[pred_is_matched]
            anchor_boxes = anchor_boxes[pred_is_matched]

            # Duplicate targets as many times as boxes assigned, tensor has size of finally assigned boxes
            image_targets = image_targets[pred_matched_target_idxs]

            for layer_idx in range(num_fpn_heads):
                # Split the selected boxes by layer (all matching lists are lists of tensors here)
                is_this_layer = layer_idxs == layer_idx
                matching_image_idxs[layer_idx].append(image_idxs[is_this_layer])
                matching_anchor_idxs[layer_idx].append(anchor_box_idxs[is_this_layer])
                matching_grid_j[layer_idx].append(grid_j[is_this_layer])
                matching_grid_i[layer_idx].append(grid_i[is_this_layer])
                matching_targets[layer_idx].append(image_targets[is_this_layer])
                matching_anchor_boxes[layer_idx].append(anchor_boxes[is_this_layer])

        # At the end, not in image by image loop
        for layer_idx in range(num_fpn_heads):
            # If any box assigned from the layer
            if matching_targets[layer_idx] != []:
                # Concatenate boxes for each image in this layer (that is why we needed bs vecs)
                matching_image_idxs[layer_idx] = torch.cat(matching_image_idxs[layer_idx], dim=0)
                matching_anchor_idxs[layer_idx] = torch.cat(matching_anchor_idxs[layer_idx], dim=0)
                matching_grid_j[layer_idx] = torch.cat(matching_grid_j[layer_idx], dim=0)
                matching_grid_i[layer_idx] = torch.cat(matching_grid_i[layer_idx], dim=0)
                matching_targets[layer_idx] = torch.cat(matching_targets[layer_idx], dim=0)
                matching_anchor_boxes[layer_idx] = torch.cat(matching_anchor_boxes[layer_idx], dim=0)
            else:
                matching_image_idxs[layer_idx] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )
                matching_anchor_idxs[layer_idx] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )
                matching_grid_j[layer_idx] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )
                matching_grid_i[layer_idx] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )
                matching_targets[layer_idx] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )
                matching_anchor_boxes[layer_idx] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )

        return (
            matching_image_idxs,
            matching_anchor_idxs,
            matching_grid_j,
            matching_grid_i,
            matching_targets,
            matching_anchor_boxes,
        )

    def find_predicted_boxes(self, fpn_heads_outputs, targets, **kwargs):
        return self.build_targets(fpn_heads_outputs, targets, **kwargs)

    def build_targets(self, fpn_heads_outputs, targets, imgs, n_anchor_per_gt=3):
        """Start with selected boxes as regular loss, but then, for each image, treat it as an OTA
        problem and select less boxes based on cost. Return those."""

        anchor_boxes_per_layer, _ = self.find_center_prior(fpn_heads_outputs, targets, n_anchors_per_target=n_anchor_per_gt)

        num_fpn_heads = len(fpn_heads_outputs)
        matched_anchor_boxes_per_layer = [[] for _ in range(num_fpn_heads)]
        matched_targets_per_layer = [[] for _ in range(num_fpn_heads)]

        num_images_in_batch = fpn_heads_outputs[0].shape[0]
        for image_idx in range(num_images_in_batch):

            is_this_image = targets[:, TargetIdx.IMG_IDX] == image_idx
            image_targets = targets[is_this_image]
            image_size = imgs[image_idx].shape[1]

            num_image_targets = image_targets.shape[0]
            if num_image_targets == 0:
                continue

            image_preds_per_layer = []
            image_anchor_boxes_per_layer = []
            layer_idxs_per_layer = []

            # For each fpn head
            for layer_idx, fpn_head_outputs in enumerate(fpn_heads_outputs):

                is_this_image = anchor_boxes_per_layer[layer_idx][:, AnchorIdx.IMG_IDX] == image_idx
                image_anchor_boxes = anchor_boxes_per_layer[layer_idx][is_this_image, :]
                image_anchor_boxes_per_layer.append(image_anchor_boxes)

                output_selector = image_anchor_boxes[:, [AnchorIdx.IMG_IDX, AnchorIdx.ANCHOR_IDX, AnchorIdx.COL, AnchorIdx.ROW]].long().T.tolist()
                selected_outputs = fpn_head_outputs[output_selector]

                image_preds = transform_model_outputs_into_predictions(selected_outputs)
                image_preds_per_layer.append(image_preds)

                layer_idxs = torch.full([image_anchor_boxes.shape[0]], fill_value=layer_idx)
                layer_idxs_per_layer.append(layer_idxs)

            layer_idxs = torch.cat(layer_idxs_per_layer, dim=0)
            image_anchor_boxes = torch.cat(image_anchor_boxes_per_layer, dim=0)
            image_preds = torch.cat(image_preds_per_layer, dim=0)
            num_image_preds = image_preds.shape[0]
            if num_image_preds == 0:
                continue

            image_targets_xywh = image_targets[:, [TargetIdx.CX, TargetIdx.CY, TargetIdx.W, TargetIdx.H]]
            image_targets_xywh_img_coords = image_targets_xywh * image_size
            image_targets_xyxy = xywh2xyxy(image_targets_xywh_img_coords)

            image_preds[..., PredIdx.CX] += image_anchor_boxes[..., AnchorIdx.ROW]
            image_preds[..., PredIdx.CY] += image_anchor_boxes[..., AnchorIdx.COL]
            image_preds[..., [PredIdx.W, PredIdx.H]] *= image_anchor_boxes[..., [AnchorIdx.W, AnchorIdx.H]]
            image_preds_xywh = image_preds[..., [PredIdx.CX, PredIdx.CY, PredIdx.W, PredIdx.H]]
            image_preds_xywh_img_coords = image_preds_xywh * self.stride[layer_idxs][:, None]
            image_preds_xyxy = xywh2xyxy(image_preds_xywh_img_coords)

            pair_wise_iou = box_iou(image_targets_xyxy, image_preds_xyxy)
            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            # Calculate dynamic K per target as defined in the OTA paper
            k = min(self.min_for_top_k, pair_wise_iou.shape[1])
            top_k_ious, _ = torch.topk(pair_wise_iou, k, dim=1)
            dynamic_ks = torch.clamp(top_k_ious.sum(dim=1).int(), min=1)

            target_class_probs = (
                F.one_hot(image_targets[:, TargetIdx.CLS_ID].long(), self.nc).float()
                .unsqueeze(dim=1).repeat(1, num_image_preds, 1)
            )
            # TODO: Find why they do geometric mean (and test if inplace is needed)
            pred_class_scores = image_preds[:, PredIdx.OBJ+1:]
            pred_objectness = image_preds[:, [PredIdx.OBJ]]
            pred_class_probs = (
                (pred_class_scores.sigmoid_() * pred_objectness.sigmoid_()).sqrt_()
                .unsqueeze(dim=0).repeat(num_image_targets, 1, 1)
            )

            pair_wise_cls_loss = F.binary_cross_entropy(pred_class_probs, target_class_probs, reduction="none").sum(-1)

            # TODO: This 3.0 is probably a constant that should be named
            cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss  # num_targets x num_preds

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
            layer_idxs = layer_idxs[pred_is_matched]
            image_matched_anchor_boxes = image_anchor_boxes[pred_is_matched]

            # Duplicate targets as many times as boxes assigned, tensor has size of finally assigned boxes
            image_matched_targets = image_targets[pred_matched_target_idxs]

            for layer_idx in range(num_fpn_heads):
                is_this_layer = layer_idxs == layer_idx
                matched_targets_per_layer[layer_idx].append(image_matched_targets[is_this_layer])
                matched_anchor_boxes_per_layer[layer_idx].append(image_matched_anchor_boxes[is_this_layer])

        # At the end, not in image by image loop
        for layer_idx in range(num_fpn_heads):
            # If any box assigned from the layer
            if matched_targets_per_layer[layer_idx]:
                # Note on matrix, rows are y, cols are x (hence 3,2 are inverted)
                grid_size = torch.tensor(fpn_heads_outputs[layer_idx].shape)[[3, 2]]
                layer_targets = torch.cat(matched_targets_per_layer[layer_idx], dim=0)
                layer_anchor_boxes = torch.cat(matched_anchor_boxes_per_layer[layer_idx], dim=0)

                layer_targets[:, [TargetIdx.CX,TargetIdx.W]] *= grid_size[0]
                layer_targets[:, [TargetIdx.CY,TargetIdx.H]] *= grid_size[1]
                layer_targets[:, [TargetIdx.CX,TargetIdx.CY]] -= layer_anchor_boxes[:, [AnchorIdx.ROW, AnchorIdx.COL]]

                matched_targets_per_layer[layer_idx] = layer_targets
                matched_anchor_boxes_per_layer[layer_idx] = layer_anchor_boxes

                # matched_targets_per_layer[layer_idx] = torch.cat(
                #     matched_targets_per_layer[layer_idx], dim=0
                # )
                # matched_anchor_boxes_per_layer[layer_idx] = torch.cat(
                #     matched_anchor_boxes_per_layer[layer_idx], dim=0
                # )
            else:
                matched_targets_per_layer[layer_idx] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )
                matched_anchor_boxes_per_layer[layer_idx] = torch.tensor(
                    [], device=targets.device, dtype=torch.int64
                )
        return matched_anchor_boxes_per_layer, matched_targets_per_layer



class ComputeYolov7LossAuxOTA(ComputeYolov7LossOTA):

    def __init__(self, model, autobalance=False):
        super().__init__(model, autobalance)
        self.min_for_top_k = 20 # Waiting understanding for better naming

    def compute_losses(self, p, targets, imgs, lcls, lbox, lobj, device, **kwargs):
        (
            bs_aux,
            as_aux_,
            gjs_aux,
            gis_aux,
            targets_aux,
            anchors_aux,
        ) = self.build_targets(p[: self.nl], targets, imgs, n_anchor_per_gt=5)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(
            p[: self.nl], targets, imgs
        )
        pre_gen_gains_aux = [
            torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[: self.nl]
        ]
        pre_gen_gains = [
            torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[: self.nl]
        ]

        # Losses
        for i in range(self.nl):  # layer index, layer predictions
            pi = p[i]
            pi_aux = p[i + self.nl]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            b_aux, a_aux, gj_aux, gi_aux = (
                bs_aux[i],
                as_aux_[i],
                gjs_aux[i],
                gis_aux[i],
            )  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tobj_aux = torch.zeros_like(pi_aux[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(
                    pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True
                )  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(
                    0
                ).type(
                    tobj.dtype
                )  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            n_aux = b_aux.shape[0]  # number of targets
            if n_aux:
                ps_aux = pi_aux[
                    b_aux, a_aux, gj_aux, gi_aux
                ]  # prediction subset corresponding to targets
                grid_aux = torch.stack([gi_aux, gj_aux], dim=1)
                pxy_aux = ps_aux[:, :2].sigmoid() * 2.0 - 0.5
                # pxy_aux = ps_aux[:, :2].sigmoid() * 3. - 1.
                pwh_aux = (ps_aux[:, 2:4].sigmoid() * 2) ** 2 * anchors_aux[i]
                pbox_aux = torch.cat((pxy_aux, pwh_aux), 1)  # predicted box
                selected_tbox_aux = targets_aux[i][:, 2:6] * pre_gen_gains_aux[i]
                selected_tbox_aux[:, :2] -= grid_aux
                iou_aux = bbox_iou(
                    pbox_aux.T, selected_tbox_aux, x1y1x2y2=False, CIoU=True
                )  # iou(prediction, target)
                lbox += 0.25 * (1.0 - iou_aux).mean()  # iou loss

                # Objectness
                tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = (
                    1.0 - self.gr
                ) + self.gr * iou_aux.detach().clamp(0).type(
                    tobj_aux.dtype
                )  # iou ratio

                # Classification
                selected_tcls_aux = targets_aux[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_aux = torch.full_like(
                        ps_aux[:, 5:], self.cn, device=device
                    )  # targets
                    t_aux[range(n_aux), selected_tcls_aux] = self.cp
                    lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)
            lobj += (
                obji * self.balance[i] + 0.25 * obji_aux * self.balance[i]
            )  # obj loss
            if self.autobalance:
                self.balance[i] = (
                    self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                )
        return tobj
