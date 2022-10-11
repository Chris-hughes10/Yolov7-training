from typing import List

import torch
import torchvision
from pytorch_accelerated import Trainer
from pytorch_accelerated.callbacks import TrainerCallback
from torch import Tensor

from yolov7.models.yolo import (
    process_yolov7_outputs,
    scale_bboxes_to_original_image_size,
)


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
        eval_loss_func,
        optimizer,
        callbacks,
        filter_eval_predictions_fn=None,
    ):
        super().__init__(
            model=model, loss_func=loss_func, optimizer=optimizer, callbacks=callbacks
        )
        self.eval_loss_func = eval_loss_func
        self.filter_eval_predictions = filter_eval_predictions_fn

    def training_run_start(self):
        self.loss_func.BCEcls.to(self.device)
        self.loss_func.BCEobj.to(self.device)
        self.loss_func.anchors = self.loss_func.anchors.to(self.device)

        self.eval_loss_func.BCEcls.to(self.device)
        self.eval_loss_func.BCEobj.to(self.device)
        self.eval_loss_func.anchors = self.eval_loss_func.anchors.to(self.device)

    def evaluation_run_start(self):
        self.loss_func.BCEcls.to(self.device)
        self.loss_func.BCEobj.to(self.device)
        self.loss_func.anchors = self.loss_func.anchors.to(self.device)

        self.eval_loss_func.BCEcls.to(self.device)
        self.eval_loss_func.BCEobj.to(self.device)
        self.eval_loss_func.anchors = self.eval_loss_func.anchors.to(self.device)

    def calculate_train_batch_loss(self, batch) -> dict:
        images, labels = batch[0], batch[1]

        model_outputs = self.model(images)
        loss, loss_items = self.loss_func(p=model_outputs, targets=labels, imgs=images)

        return {
            "loss": loss,
            "model_outputs": model_outputs,
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
            model_outputs = self.model(images)

            inference_outputs, rpn_outputs = model_outputs
            val_loss, loss_items = self.eval_loss_func(p=rpn_outputs, targets=labels)
            preds = process_yolov7_outputs(
                model_outputs,
            )

            if self.filter_eval_predictions is not None:
                preds = self.filter_eval_predictions(preds)

            resized_image_sizes = torch.as_tensor(
                images.shape[2:], device=original_image_sizes.device
            )[None].repeat(len(inference_outputs), 1)

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
            "model_outputs": model_outputs,
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
