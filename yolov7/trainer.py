from typing import List

import torch
import torchvision
from pytorch_accelerated import Trainer
from pytorch_accelerated.callbacks import TrainerCallback
from torch import Tensor

from yolov7.models.yolo import (
    scale_bboxes_to_original_image_size,
)


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


class DisableAugmentationCallback(TrainerCallback):
    def __init__(self, no_aug_epochs):
        self.no_aug_epochs = no_aug_epochs

    def on_train_epoch_start(self, trainer, **kwargs):
        if (
            trainer.run_history.current_epoch
            == trainer.run_config.num_epochs - self.no_aug_epochs
        ):
            trainer.print("Disabling Mosaic Augmentation")
            trainer.train_dataset.ds.disable()


def filter_eval_predictions(
    predictions: Tensor, confidence_threshold: float = 0.2, nms_threshold: float = 0.65
) -> List[Tensor]:
    nms_preds = []
    for pred in predictions:
        pred = pred[pred[:, 4] > confidence_threshold]

        nms_idx = torchvision.ops.batched_nms(
            boxes=pred[:, :4],
            scores=pred[:, 4],
            idxs=pred[:, 5],
            iou_threshold=nms_threshold,
        )
        nms_preds.append(pred[nms_idx])

    return nms_preds


class Yolov7Trainer(Trainer):
    YOLO7_PADDING_VALUE = -2.0

    def __init__(
        self,
        model,
        loss_func,
        optimizer,
        callbacks,
        filter_eval_predictions_fn=None,
    ):
        super().__init__(
            model=model, loss_func=loss_func, optimizer=optimizer, callbacks=callbacks
        )
        self.filter_eval_predictions = filter_eval_predictions_fn

    def training_run_start(self):
        self.loss_func.to(self.device)

    def evaluation_run_start(self):
        self.loss_func.to(self.device)

    def train_epoch_start(self):
        super().train_epoch_start()
        self.loss_func.train()

    def eval_epoch_start(self):
        super().eval_epoch_start()
        self.loss_func.eval()

    def calculate_train_batch_loss(self, batch) -> dict:
        images, labels = batch[0], batch[1]

        fpn_heads_outputs = self.model(images)
        loss, _ = self.loss_func(
            fpn_heads_outputs=fpn_heads_outputs, targets=labels, images=images
        )

        return {
            "loss": loss,
            "model_outputs": fpn_heads_outputs,
            "batch_size": images.size(0),
        }

    def calculate_eval_batch_loss(self, batch) -> dict:
        with torch.no_grad():
            images, labels, image_idxs, original_image_sizes = (
                batch[0],
                batch[1],
                batch[2],
                batch[3].cpu(),
            )
            fpn_heads_outputs = self.model(images)
            val_loss, _ = self.loss_func(
                fpn_heads_outputs=fpn_heads_outputs, targets=labels
            )

            preds = self.model.postprocess(fpn_heads_outputs, conf_thres=0.001)

            if self.filter_eval_predictions is not None:
                preds = self.filter_eval_predictions(preds)

            resized_image_sizes = torch.as_tensor(
                images.shape[2:], device=original_image_sizes.device
            )[None].repeat(len(preds), 1)

        formatted_predictions = self.get_formatted_preds(
            image_idxs, preds, original_image_sizes, resized_image_sizes
        )

        gathered_predictions = (
            self.gather(formatted_predictions, padding_value=self.YOLO7_PADDING_VALUE)
            .detach()
            .cpu()
        )

        return {
            "loss": val_loss,
            "model_outputs": fpn_heads_outputs,
            "predictions": gathered_predictions,
            "batch_size": images.size(0),
        }

    def get_formatted_preds(
        self, image_idxs, preds, original_image_sizes, resized_image_sizes
    ):
        """
        scale bboxes to original image dimensions, and associate image idx with predictions
        :param image_idxs:
        :param preds:
        :param original_image_sizes:
        :param resized_image_sizes:
        :return:
        """
        formatted_preds = []
        for i, (image_idx, image_preds) in enumerate(zip(image_idxs, preds)):
            # x1, y1, x2, y2, score, class_id, image_idx
            formatted_preds.append(
                torch.cat(
                    (
                        scale_bboxes_to_original_image_size(
                            image_preds[:, :4],
                            resized_hw=resized_image_sizes[i],
                            original_hw=original_image_sizes[i],
                            is_padded=True,
                        ),
                        image_preds[:, 4:],
                        image_idx.repeat(image_preds.shape[0])[None].T,
                    ),
                    1,
                )
            )

        if not formatted_preds:
            # create placeholder so that it can be gathered across processes
            stacked_preds = torch.tensor(
                [self.YOLO7_PADDING_VALUE] * 7, device=self.device
            )[None]
        else:
            stacked_preds = torch.vstack(formatted_preds)

        return stacked_preds
