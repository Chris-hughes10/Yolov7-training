import torch
import torchvision.ops
from pytorch_accelerated import Trainer
from pytorch_accelerated.callbacks import TrainerCallback

import pandas as pd
from torch import Tensor


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


class Yolov7Trainer(Trainer):
    YOLO7_PADDING_VALUE = -2.0

    def __init__(
        self,
        model,
        loss_func,
        eval_loss_func,
        optimizer,
        callbacks,
        eval_image_idx_to_id_lookup,
    ):
        super().__init__(
            model=model, loss_func=loss_func, optimizer=optimizer, callbacks=callbacks
        )
        self.eval_loss_func = eval_loss_func
        self.eval_predictions = []
        self.eval_image_idx_to_id_lookup = eval_image_idx_to_id_lookup

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
        loss, loss_items = self.loss_func(model_outputs, labels, images)

        return {
            "loss": loss,
            "model_outputs": model_outputs,
            "batch_size": images.size(0),
        }


    def calculate_eval_batch_loss(self, batch) -> dict:
        with torch.no_grad():
            images, labels, image_idxs = batch[0], batch[1], batch[2]
            model_outputs = self.model(images)
            inference_outputs, rpn_outputs = model_outputs
            val_loss, loss_items = self.eval_loss_func(rpn_outputs, labels)
            preds = self.model.process_outputs(model_outputs)

            nms_preds = []

            for pred in preds:
                nms_idx = torchvision.ops.batched_nms(
                    boxes=pred[:, :4],
                    scores=pred[:, 4],
                    idxs=pred[:, 5],
                    iou_threshold=0.1,
                )
                nms_preds.append(pred[nms_idx])

            preds = nms_preds

        formatted_predictions = (
            self.get_formatted_preds(image_idxs, preds).detach().cpu()
        )
        formatted_targets = self.get_formatted_targets(labels, image_idxs, images)

        gathered_predictions = self.gather(
            formatted_predictions, padding_value=self.YOLO7_PADDING_VALUE
        )
        gathered_targets = (
            self.gather(formatted_targets, padding_value=self.YOLO7_PADDING_VALUE)
            .detach()
            .cpu()
        )

        return {
            "loss": val_loss,
            "model_outputs": model_outputs,
            "predictions": gathered_predictions,
            "targets": gathered_targets,
            "batch_size": images.size(0),
        }

    def get_formatted_preds(self, image_idxs, preds):
        formatted_preds = []
        for image_idx, image_preds in zip(image_idxs, preds):
            # x1, y1, x2, y2, score, class_id, image_idx
            formatted_preds.append(
                torch.cat(
                    (
                        image_preds,
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

    def get_formatted_targets(self, labels, image_idxs, images):
        formatted_targets = []
        for im_i, image_idx in enumerate(image_idxs):
            image_labels = labels[labels[:, 0] == im_i][:, 1:].clone()

            # denormalize
            image_labels[:, [1, 3]] = image_labels[:, [1, 3]] * images.shape[-2]
            image_labels[:, [2, 4]] = image_labels[:, [2, 4]] * images.shape[-1]
            xyxy_labels = torchvision.ops.box_convert(
                image_labels[:, 1:], "cxcywh", "xyxy"
            )
            formatted_targets.append(
                # cx, cy, w, h, class_id, image_idx
                torch.cat(
                    (
                        xyxy_labels,
                        image_labels[:, 0][None].T,
                        image_idx.repeat(image_labels.shape[0])[None].T,
                    ),
                    1,
                )
            )

        if not formatted_targets:
            # create placeholder so that it can be gathered across processes
            stacked_targets = torch.tensor(
                [self.YOLO7_PADDING_VALUE] * 6, device=self.device
            )[None]
        else:
            stacked_targets = torch.vstack(formatted_targets)

        return stacked_targets
