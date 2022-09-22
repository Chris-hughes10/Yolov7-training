import torch
import torchvision.ops
from pytorch_accelerated import Trainer
from pytorch_accelerated.callbacks import TrainerCallback

import pandas as pd

from yolov7.model_factory import process_yolov7_outputs


class DisableAugmentationCallback(TrainerCallback):
    def __init__(self, no_aug_epochs):
        self.no_aug_epochs = no_aug_epochs

    def on_train_epoch_start(self, trainer, **kwargs):
        if trainer.run_history.current_epoch == trainer.run_config.num_epochs - self.no_aug_epochs:
            trainer.print("Disabling Mosaic Augmentation")
            trainer.train_dataset.ds.disable()

class Yolov7Trainer(Trainer):
    YOLO7_PADDING_VALUE = -2.0

    def __init__(self, model, loss_func, eval_loss_func, optimizer, callbacks, eval_image_idx_to_id_lookup):
        super().__init__(model=model, loss_func=loss_func, optimizer=optimizer, callbacks=callbacks)
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

    def eval_epoch_start(self):
        super(Yolov7Trainer, self).eval_epoch_start()
        self.eval_predictions = []
        self.eval_targets = []
        self.idxs_seen = set()
        self.preds_df = None

    def calculate_eval_batch_loss(self, batch) -> dict:
        with torch.no_grad():
            images, labels, image_idxs = batch[0], batch[1], batch[2]
            model_outputs = self.model(images)
            inference_outputs, rpn_outputs = model_outputs
            val_loss, loss_items = self.eval_loss_func(rpn_outputs, labels)
            preds = process_yolov7_outputs(model_outputs)

            nms_preds = []

            for pred in preds:
                nms_idx = torchvision.ops.batched_nms(boxes=pred[:, :4],
                                                      scores=pred[:, 4],
                                                      idxs=pred[:, 5],
                                                      iou_threshold=0.1)
                nms_preds.append(pred[nms_idx])

            preds = nms_preds

            # remove any image_idx that has already been seen
            # this can arise from distributed training where batch size does not evenly divide dataset
            seen_idx_mask = torch.as_tensor(
                [False if idx not in self.idxs_seen else True for idx in image_idxs.tolist()]
            )
            
            if seen_idx_mask.all():
                # handle where all have been seen
                image_idxs = torch.as_tensor([int(self.YOLO7_PADDING_VALUE)], device=self.device)
                preds = []
                labels = []
            
            elif seen_idx_mask.any(): # at least one True
                unseen_idxs_indices = (~seen_idx_mask).nonzero().tolist()[0]
                image_idxs = image_idxs[unseen_idxs_indices]
                preds = [preds[i] for i in unseen_idxs_indices]

                if not image_idxs:
                    image_idxs = torch.as_tensor(
                        [int(self.YOLO7_PADDING_VALUE)], device=self.device
                    ).repeat(batch[2].shape[0])

            gathered_predictions = self.get_formatted_preds(image_idxs, preds)
            self.eval_predictions.extend(gathered_predictions.detach().cpu().tolist())

        formatted_targets = []
        for im_i, (image_idx, seen_idx) in enumerate(zip(image_idxs, seen_idx_mask)):
            if not seen_idx:
                image_labels = labels[labels[:, 0] == im_i][:, 1:].clone()

                # denormalize
                image_labels[:, [1, 3]] = image_labels[:, [1, 3]] * images.shape[-2]
                image_labels[:, [2, 4]] = image_labels[:, [2, 4]] * images.shape[-1]
                xyxy_labels = torchvision.ops.box_convert(image_labels[:, 1:], 'cxcywh', 'xyxy')
                formatted_targets.append(
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
            stacked_targets = torch.tensor([self.YOLO7_PADDING_VALUE] * 6, device=self.device)[None]
        else:
            stacked_targets = torch.vstack(formatted_targets)

        padded_stacked_targets = self._accelerator.pad_across_processes(
            stacked_targets, pad_index=self.YOLO7_PADDING_VALUE
        )

        gathered_targets = self.gather(padded_stacked_targets)

        if len(gathered_targets) > 0:
            gathered_targets = gathered_targets[
                gathered_targets[:, 0] != self.YOLO7_PADDING_VALUE
                ]
        self._accelerator.wait_for_everyone()
        self.eval_targets.extend(gathered_targets.detach().cpu().tolist())
        
        padded_indices = self._accelerator.pad_across_processes(
            image_idxs, pad_index=int(self.YOLO7_PADDING_VALUE)
        )
        gathered_indices = self.gather(padded_indices)

        gathered_indices = gathered_indices[
            gathered_indices != int(self.YOLO7_PADDING_VALUE)
        ].tolist()

        self.idxs_seen.update(gathered_indices)

        return {
            "loss": val_loss,
            "model_outputs": model_outputs,
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
            stacked_preds = torch.tensor([self.YOLO7_PADDING_VALUE] * 7, device=self.device)[None]
        else:
            stacked_preds = torch.vstack(formatted_preds)

        padded_stacked_preds = self._accelerator.pad_across_processes(
            stacked_preds, pad_index=self.YOLO7_PADDING_VALUE
        )

        gathered_predictions = self.gather(padded_stacked_preds)

        if len(gathered_predictions) > 0:
            gathered_predictions = gathered_predictions[
                gathered_predictions[:, 0] != self.YOLO7_PADDING_VALUE
                ]
        self._accelerator.wait_for_everyone()

        return gathered_predictions

    def eval_epoch_end(self):

        preds_df = pd.DataFrame(torch.as_tensor(self.eval_predictions),
                                columns=['xmin', 'ymin', 'xmax', 'ymax', 'score', 'class_id', 'image_idx'])
        preds_df['image_id'] = preds_df.image_idx.map(self.eval_image_idx_to_id_lookup)
        self.preds_df = preds_df

        targets_df = pd.DataFrame(torch.as_tensor(self.eval_targets),
                                  columns=['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'image_idx'])
        targets_df['image_id'] = targets_df.image_idx.map(self.eval_image_idx_to_id_lookup)
        # targets_df['class_id'] = 2
        self.targets_df = targets_df

