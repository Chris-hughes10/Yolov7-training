import os
from functools import partial
from pathlib import Path

import torch
from func_to_script import script
from pytorch_accelerated import notebook_launcher
from pytorch_accelerated.callbacks import get_default_callbacks, SaveBestModelCallback
from pytorch_accelerated.schedulers import CosineLrScheduler
from torch import nn

from example.data import DatasetAdaptor, load_cars_df
from yolov7 import create_yolov7_model
from yolov7.dataset import (
    create_base_transforms,
    Yolov7Dataset,
    create_yolov7_transforms,
    yolov7_collate_fn,
)
from yolov7.evaluation import CalculateMetricsCallback
from yolov7.loss_factory import create_yolov7_loss_orig
from yolov7.mosaic import MosaicMixupDataset, create_post_mosaic_transform
from yolov7.trainer import Yolov7Trainer, filter_eval_predictions
from yolov7.utils import SaveFirstBatchCallback


@script
def main(
    data_path: str = "/home/chris/Downloads/data",
    image_size: int = 640,
    pretrained: bool = True,
    num_epochs: int = 30,
):

    # pretrained = False
    num_epochs = 100

    data_path = Path(data_path)
    images_path = data_path / "training_images"
    annotations_file_path = data_path / "annotations.csv"

    train_df, valid_df, lookups = load_cars_df(annotations_file_path, images_path)

    train_ds = DatasetAdaptor(
        images_path,
        train_df,
        transforms=create_base_transforms(image_size)
        # transforms=create_base_transforms(1280)
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

    conv_weights = {
        v.weight
        for k, v in model.named_modules()
        if (
            hasattr(v, "weight")
            and isinstance(v.weight, nn.Parameter)
            and not isinstance(v, nn.BatchNorm2d)
        )
    }

    other_params = [p for p in model.parameters() if p not in conv_weights]

    lr = 0.01
    optimizer = torch.optim.SGD(other_params, lr=lr, momentum=0.937, nesterov=True)

    # lr = 0.001
    # optimizer = torch.optim.AdamW(other_params, lr=lr)

    loss_func = create_yolov7_loss_orig(model, aux_loss=False)

    cooldown_epochs = 5

    batch_size = 8

    trainer = Yolov7Trainer(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        eval_loss_func=create_yolov7_loss_orig(model, ota_loss=False),
        filter_eval_predictions_fn=partial(
            filter_eval_predictions, confidence_threshold=0.01, nms_threshold=0.3
        ),
        callbacks=[
            CalculateMetricsCallback.create_from_targets_df(
                targets_df=valid_df.query("has_annotation == True")[
                    ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id"]
                ],
                image_ids=set(valid_df.image_id.unique()),
                iou_threshold=0.2,
            ),
            SaveBestModelCallback(watch_metric="map", greater_is_better=True),
            SaveFirstBatchCallback("./batches", num_images_per_batch=3),
            *get_default_callbacks(progress_bar=True),
        ],
    )

    nbs = 64  # nominal batch size
    total_batch_size = (
        batch_size * trainer._accelerator.num_processes
    )  # batch size across all processes
    num_accumulate_steps = max(
        round(nbs / total_batch_size), 1
    )  # calculate num accum steps
    weight_decay = 0.0005  # default
    weight_decay *= total_batch_size * num_accumulate_steps / nbs  # scale weight_decay

    optimizer.add_param_group(
        {"params": list(conv_weights), "weight_decay": weight_decay}
    )  # add pg1 with weight_decay

    trainer.train(
        num_epochs=num_epochs,
        train_dataset=train_yds,
        train_dataloader_kwargs={"num_workers": 0},
        eval_dataset=eval_yds,
        per_device_batch_size=batch_size,
        create_scheduler_fn=CosineLrScheduler.create_scheduler_fn(
            num_warmup_epochs=5, num_cooldown_epochs=cooldown_epochs
        ),
        collate_fn=yolov7_collate_fn,
        gradient_accumulation_steps=num_accumulate_steps,
    )


if __name__ == "__main__":
    os.environ["mixed_precision"] = "fp16"
    # main()
    notebook_launcher(main, num_processes=2)
