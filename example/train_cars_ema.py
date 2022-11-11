from copy import deepcopy
import math
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from func_to_script import script
from PIL import Image
from pytorch_accelerated.callbacks import (
    SaveBestModelCallback,
    get_default_callbacks,
    CallbackHandler,
    ProgressBarCallback
)
from pytorch_accelerated.schedulers import CosineLrScheduler
from torch.utils.data import Dataset

from yolov7 import create_yolov7_model
from yolov7.dataset import (
    Yolov7Dataset,
    create_base_transforms,
    create_yolov7_transforms,
    yolov7_collate_fn,
)
from yolov7.evaluation import CalculateMeanAveragePrecisionCallback
from yolov7.loss_factory import create_yolov7_loss
from yolov7.mosaic import MosaicMixupDataset, create_post_mosaic_transform
from yolov7.trainer import Yolov7Trainer, filter_eval_predictions
from yolov7.utils import SaveBatchesCallback

class ModelEma(nn.Module):
    """
    Maintains a moving average of everything in the model state_dict (parameters and buffers), based on the ideas
    from https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage.

    This class maintains a copy of the model that we are training. However,
    rather than updating all of the parameters of this model after every update step,
    we set these parameters using a linear combination of the existing parameter values and the updated values

    .. Note:: It is important to note that this class is sensitive to where it is initialised.
        During distributed training, it should be applied before before the conversion to :class:`~torch.nn.SyncBatchNorm`
        takes place and before the :class:`torch.nn.parallel.DistributedDataParallel` wrapper is used!
    """

    def __init__(self, model, decay=0.9999):
        """
        Create an instance of :class:`torch.nn.Module` to maintain an exponential moving average of the weights of
        the given model.

        This is done using the following formula:

        `updated_EMA_model_weights = decay * EMA_model_weights + (1. — decay) * updated_model_weights`

        where the decay is a parameter that we set.

        :param model: the subclass of :class: `torch.nn.Module` that we are training. This is the model that will be updated in our training loop as normal.
        :param decay: the amount of decay to use, which determines how much of the previous state will be maintained. The TensorFlow documentation suggests that reasonable values for decay are close to 1.0, typically in the multiple-nines range: 0.999, 0.9999

        """
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.module.eval()
        self.decay = decay

    def update_fn(self, ema_model_weights, updated_model_weights):
        return (
            self.decay * ema_model_weights + (1.0 - self.decay) * updated_model_weights
        )

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                updated_v = update_fn(ema_v, model_v)
                ema_v.copy_(updated_v)

    def update(self, model):
        self._update(model, update_fn=self.update_fn)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class Yolov7ModelEma(ModelEma):

    def __init__(self, model, decay=0.9990):
        super().__init__(model, decay)
        self.num_updates = 0
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        self.decay = self.decay_fn(self.num_updates)

    def update(self, model):
        super().update(model)
        self.num_updates += 1
        self.decay = self.decay_fn(self.num_updates)



class ModelEmaCallback(SaveBestModelCallback):
    """
    A callback which maintains and saves an exponential moving average of the weights of the model that is currently
    being trained.

    This callback offers the option of evaluating the EMA model during. If enabled, this is done by running an additional
    validation after each training epoch, which will use additional GPU resources. During this additional epoch,
    only the provided callbacks will be executed.

    .. Note:: This callback is sensitive to the order that it is executed. This should be placed after any callbacks that 
        modify state (e.g. metrics) but before any callbacks that read state (e.g. loggers).

    """

    def __init__(
        self,
        decay: float = 0.99,
        evaluate_during_training: bool = True,
        save_path: str = "ema_model.pt",
        watch_metric: str = "ema_model_eval_loss_epoch",
        greater_is_better: bool = False,
        model_ema=ModelEma,
        callbacks=(),
    ):
        """
        :param decay: the amount of decay to use, which determines how much of the previous state will be maintained.
        :param evaluate_during_training: whether to evaluate the EMA model during training. If True, an additional validation epoch will be conducted after each training epoch, which will use additional GPU resources, and the best model will be saved. If False, the saved EMA model checkpoint will be updated at the end of each epoch.
        :param watch_metric: the metric used to compare model performance. This should be accessible from the trainer's run history. This is only used when ``evaluate_during_training`` is enabled.
        :param greater_is_better: whether an increase in the ``watch_metric`` should be interpreted as the model performing better.
        :param model_ema: the class which is responsible for maintaining the moving average of the model.
        :param callbacks: an iterable of callbacks that will be executed during the evaluation loop of the EMA model

        """
        super().__init__(
            save_path=save_path,
            watch_metric=watch_metric,
            greater_is_better=greater_is_better,
            reset_on_train=False,
            save_optimizer=False,
        )
        self.decay = decay
        self.ema_model = None
        self._track_prefix = "ema_model_"
        self.evaluate_during_training = evaluate_during_training
        self.model_ema_cls = model_ema
        self.callback_handler = CallbackHandler(callbacks)

    def on_training_run_start(self, trainer, **kwargs):
        self.ema_model = self.model_ema_cls(
            trainer._accelerator.unwrap_model(trainer.model), decay=self.decay
        )
        if self.evaluate_during_training:
            self.ema_model.to(trainer.device)

    def on_train_epoch_end(self, trainer, **kwargs):
        self.ema_model.update(trainer._accelerator.unwrap_model(trainer.model))

    def on_eval_epoch_end(self, trainer, **kwargs):
        if self.evaluate_during_training:
            model = trainer.model
            trainer.model = self.ema_model.module
            run_history_prefix = trainer.run_history.metric_name_prefix
            trainer_callback_handler = trainer.callback_handler

            trainer.print("Running evaluation on EMA model")

            trainer.callback_handler = self.callback_handler
            trainer.run_history.set_metric_name_prefix(self._track_prefix)
            trainer._run_eval_epoch(trainer._eval_dataloader)

            trainer.model = model
            trainer.callback_handler = trainer_callback_handler
            trainer.run_history.set_metric_name_prefix(run_history_prefix)

    def on_training_run_epoch_end(self, trainer, **kwargs):
        model = trainer.model
        trainer.model = self.ema_model.module

        if self.evaluate_during_training:
            super().on_training_run_epoch_end(trainer)
        else:
            trainer.save_checkpoint(save_path=self.save_path, save_optimizer=False)

        trainer.model = model

    def on_training_run_end(self, trainer, **kwargs):
        # Overriding, as we do not want to load the EMA model
        pass


def load_cars_df(annotations_file_path, images_path):
    all_images = sorted(set([p.parts[-1] for p in images_path.iterdir()]))
    image_id_to_image = {i: im for i, im in enumerate(all_images)}
    image_to_image_id = {v: k for k, v, in image_id_to_image.items()}

    annotations_df = pd.read_csv(annotations_file_path)
    annotations_df.loc[:, "class_name"] = "car"
    annotations_df.loc[:, "has_annotation"] = True

    # add 100 empty images to the dataset
    empty_images = sorted(set(all_images) - set(annotations_df.image.unique()))
    non_annotated_df = pd.DataFrame(list(empty_images)[:100], columns=["image"])
    non_annotated_df.loc[:, "has_annotation"] = False
    non_annotated_df.loc[:, "class_name"] = "background"

    df = pd.concat((annotations_df, non_annotated_df))

    class_id_to_label = dict(
        enumerate(df.query("has_annotation == True").class_name.unique())
    )
    class_label_to_id = {v: k for k, v in class_id_to_label.items()}

    df["image_id"] = df.image.map(image_to_image_id)
    df["class_id"] = df.class_name.map(class_label_to_id)

    file_names = tuple(df.image.unique())
    random.seed(42)
    validation_files = set(random.sample(file_names, int(len(df) * 0.2)))
    train_df = df[~df.image.isin(validation_files)]
    valid_df = df[df.image.isin(validation_files)]

    lookups = {
        "image_id_to_image": image_id_to_image,
        "image_to_image_id": image_to_image_id,
        "class_id_to_label": class_id_to_label,
        "class_label_to_id": class_label_to_id,
    }
    return train_df, valid_df, lookups


class DatasetAdaptor(Dataset):
    def __init__(
        self,
        images_dir_path,
        annotations_dataframe,
        transforms=None,
    ):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_dataframe
        self.transforms = transforms

        self.image_idx_to_image_id = {
            idx: image_id
            for idx, image_id in enumerate(self.annotations_df.image_id.unique())
        }
        self.image_id_to_image_idx = {
            v: k for k, v, in self.image_idx_to_image_id.items()
        }

    def __len__(self) -> int:
        return len(self.image_idx_to_image_id)

    def __getitem__(self, index):
        image_id = self.image_idx_to_image_id[index]
        image_info = self.annotations_df[self.annotations_df.image_id == image_id]
        file_name = image_info.image.values[0]
        assert image_id == image_info.image_id.values[0]

        image = Image.open(self.images_dir_path / file_name).convert("RGB")
        image = np.array(image)

        image_hw = image.shape[:2]

        if image_info.has_annotation.any():
            xyxy_bboxes = image_info[["xmin", "ymin", "xmax", "ymax"]].values
            class_ids = image_info["class_id"].values
        else:
            xyxy_bboxes = np.array([])
            class_ids = np.array([])

        if self.transforms is not None:
            transformed = self.transforms(
                image=image, bboxes=xyxy_bboxes, labels=class_ids
            )
            image = transformed["image"]
            xyxy_bboxes = np.array(transformed["bboxes"])
            class_ids = np.array(transformed["labels"])

        return image, xyxy_bboxes, class_ids, image_id, image_hw


@script
def main(
    data_path: str = "./data/cars",
    image_size: int = 640,
    pretrained: bool = True,
    num_epochs: int = 30,
):
    pretrained=False
    num_epochs=300

    data_path = Path(data_path)
    images_path = data_path / "training_images"
    annotations_file_path = data_path / "annotations.csv"

    train_df, valid_df, lookups = load_cars_df(annotations_file_path, images_path)

    train_ds = DatasetAdaptor(
        images_path, train_df, transforms=create_base_transforms(image_size)
    )
    eval_ds = DatasetAdaptor(images_path, valid_df)

    mds = MosaicMixupDataset(
        train_ds,
        apply_mixup_probability=0.15,
        post_mosaic_transforms=create_post_mosaic_transform(
            output_height=image_size, output_width=image_size
        ),
    )
    if pretrained:
        mds.disable()

    train_yds = Yolov7Dataset(
        mds,
        create_yolov7_transforms(training=True, image_size=(image_size, image_size)),
    )
    eval_yds = Yolov7Dataset(
        eval_ds,
        create_yolov7_transforms(training=False, image_size=(image_size, image_size)),
    )

    num_classes = 1

    model = create_yolov7_model(
        architecture="yolov7", num_classes=num_classes, pretrained=pretrained
    )

    param_groups = model.get_parameter_groups()

    optimizer = torch.optim.SGD(
        param_groups["other_params"], lr=0.01, momentum=0.937, nesterov=True
    )

    loss_func = create_yolov7_loss(model, image_size=image_size)

    cooldown_epochs = 5

    batch_size = 8

    calculate_map_callback = CalculateMeanAveragePrecisionCallback.create_from_targets_df(
                targets_df=valid_df.query("has_annotation == True")[
                    ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id"]
                ],
                image_ids=set(valid_df.image_id.unique()),
                iou_threshold=0.2,
            )

    trainer = Yolov7Trainer(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        filter_eval_predictions_fn=partial(
            filter_eval_predictions, confidence_threshold=0.01, nms_threshold=0.3
        ),
        callbacks=[
            calculate_map_callback,
            ModelEmaCallback(decay=0.9999,
             model_ema=Yolov7ModelEma,
             callbacks=[ProgressBarCallback, calculate_map_callback]),
            SaveBestModelCallback(watch_metric="map", greater_is_better=True),
            SaveBatchesCallback("./batches", num_images_per_batch=3),
            *get_default_callbacks(progress_bar=True),
        ],
    )

    total_batch_size = (
        batch_size * trainer._accelerator.num_processes
    )  # batch size across all processes

    nbs = 64  # nominal batch size
    num_accumulate_steps = max(
        round(nbs / total_batch_size), 1
    )  # calculate num accum steps
    weight_decay = 0.0005  # default
    weight_decay *= total_batch_size * num_accumulate_steps / nbs  # scale weight_decay

    optimizer.add_param_group(
        {"params": param_groups["conv_weights"], "weight_decay": weight_decay}
    )  # add pg1 with weight_decay

    trainer.train(
        num_epochs=num_epochs,
        train_dataset=train_yds,
        eval_dataset=eval_yds,
        per_device_batch_size=batch_size,
        create_scheduler_fn=CosineLrScheduler.create_scheduler_fn(
            num_warmup_epochs=5,
            num_cooldown_epochs=cooldown_epochs,
            k_decay=2,
        ),
        collate_fn=yolov7_collate_fn,
        gradient_accumulation_steps=num_accumulate_steps,
    )


if __name__ == "__main__":
    os.environ["mixed_precision"] = "fp16"
    main()
