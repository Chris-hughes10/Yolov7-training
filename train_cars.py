import os
from pathlib import Path
import random

import timm.optim
import torch
from pytorch_accelerated.callbacks import get_default_callbacks
from pytorch_accelerated.schedulers import CosineLrScheduler
from torch import nn

from custom.data import DatasetAdaptor, load_cars_df
from custom.datasets import MosaicMixupDataset
from custom.calculate_map import CalculateMetricsCallback
from custom.yolo7_utils.dataset import create_base_transforms, Yolov7Dataset, create_yolov7_transforms, \
    yolov7_collate_fn
from custom.yolo7_utils.loss import create_loss
from custom.yolo7_utils.model import create_yolov7_model
from custom.yolo7_utils.trainer import Yolov7Trainer
from custom_train import DisableAugmentationCallback


def main():
    data_path = '/home/chris/notebooks/yolov7/cars/data'
    data_path = Path(data_path)
    images_path = data_path / "training_images"
    annotations_file_path = data_path / "annotations.csv"
    yolov7_config_path = Path('/home/chris/notebooks/yolov7/custom/yolo7-cfg')

    train_df, valid_df, lookups= load_cars_df(annotations_file_path, images_path)

    label_to_class_id = {'car': 0}


    train_ds = DatasetAdaptor(images_path, train_df, label_to_class_id, bgr_images=False,
                              transforms=create_base_transforms(640)
                              # transforms=create_base_transforms(1280)
                              )
    eval_ds = DatasetAdaptor(images_path, valid_df, label_to_class_id, bgr_images=False)

    mds = MosaicMixupDataset(train_ds, apply_mixup_probability=0.15)
    mds.disable()

    train_yds = Yolov7Dataset(mds, create_yolov7_transforms(training=True, image_size=(640, 640)))
    eval_yds = Yolov7Dataset(eval_ds, create_yolov7_transforms(training=False, image_size=(640, 640)))

    num_classes = 1

    # model = create_yolov7_model(
    #     config=yolov7_config_path / "yolov7-e6e.yaml",
    #     state_dict_path=yolov7_config_path / "yolov7-e6e_training_state_dict.pt",
    #     num_classes=num_classes,
    # )

    model = create_yolov7_model(
        config=yolov7_config_path / "yolov7.yaml",
        state_dict_path=yolov7_config_path / "yolov7_training_state_dict.pt",
        num_classes=num_classes,
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

    momentum = 0.937
    # lr = 0.001
    lr = 0.01

    optimizer = torch.optim.SGD(other_params, lr=lr, momentum=momentum, nesterov=True)
    # optimizer = timm.optim.AdamW(other_params, lr=lr, weight_decay=0)

    loss_func = create_loss(model, aux_loss=False)

    cooldown_epochs = 5

    batch_size=8

    trainer = Yolov7Trainer(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        eval_loss_func=create_loss(model, ota_loss=False),
        eval_image_idx_to_id_lookup=eval_ds.image_idx_to_image_id,
        callbacks=[
            DisableAugmentationCallback(no_aug_epochs=cooldown_epochs),

            CalculateMetricsCallback(targets_df=valid_df.query('has_annotation == True'), eval_image_ids=set(valid_df.image_id.unique())),
            *get_default_callbacks(progress_bar=True),
        ],
    )

    nbs = 64  # nominal batch size
    total_batch_size = batch_size * trainer._accelerator.num_processes  # batch size across all processes
    num_accumulate_steps = max(round(nbs / total_batch_size), 1)  # calculate num accum steps
    weight_decay = 0.0005  # default
    weight_decay *= total_batch_size * num_accumulate_steps / nbs  # scale weight_decay

    optimizer.add_param_group(
        {"params": list(conv_weights), "weight_decay": weight_decay}
    )  # add pg1 with weight_decay

    trainer.train(
        num_epochs=20,
        train_dataset=train_yds,
        eval_dataset=eval_yds,
        per_device_batch_size=batch_size,
        create_scheduler_fn=CosineLrScheduler.create_scheduler_fn(
            num_warmup_epochs=0, num_cooldown_epochs=cooldown_epochs
        ),
        # train_dataloader_kwargs=train_dl_kwargs,
        # eval_dataloader_kwargs={"pin_memory": False, "num_workers": 0},
        collate_fn=yolov7_collate_fn,
        gradient_accumulation_steps=num_accumulate_steps,
    )

if __name__ == '__main__':
    os.environ['mixed_precision'] = 'fp16'
    main()