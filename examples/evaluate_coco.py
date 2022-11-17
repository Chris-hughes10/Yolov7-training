from functools import partial
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torchvision
from func_to_script import script
from pytorch_accelerated.callbacks import TrainerCallback, get_default_callbacks
from torchvision.datasets.coco import CocoDetection

from yolov7 import create_yolov7_model
from yolov7.dataset import Yolov7Dataset, create_yolov7_transforms, yolov7_collate_fn
from yolov7.evaluation import CalculateMeanAveragePrecisionCallback
from yolov7.loss_factory import create_yolov7_loss
from yolov7.trainer import Yolov7Trainer, filter_eval_predictions

# fmt: off
COCO80_TO_COCO91_MAP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                        56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
                        85, 86, 87, 88, 89, 90]


# fmt: on


def coco80_to_coco91_lookup():
    return {i: v for i, v in enumerate(COCO80_TO_COCO91_MAP)}


def coco91_to_coco80_lookup():
    return {v: i for i, v in enumerate(COCO80_TO_COCO91_MAP)}


class COCOBaseDataset(CocoDetection):
    def __init__(self, img_dir, annotation_path, tfms=None):
        super().__init__(root=str(img_dir), annFile=str(annotation_path))
        self.lookup = coco91_to_coco80_lookup()
        self.targets_json = self.coco.dataset
        self.tfms = tfms

    def __getitem__(self, index):
        image_id = self.ids[index]

        image, targets = super().__getitem__(index)
        image = np.array(image)
        image_hw = image.shape[:2]
        raw_boxes = [target["bbox"] for target in targets if target["iscrowd"] == 0]

        if len(raw_boxes) == 0:
            xyxy_bboxes = np.array([])
            class_ids = np.array([])
        else:
            xyxy_bboxes = torchvision.ops.box_convert(
                torch.as_tensor(raw_boxes), "xywh", "xyxy"
            ).numpy()
            class_ids = np.array(
                [
                    self.lookup[target["category_id"]]
                    for target in targets
                    if target["iscrowd"] == 0
                ]
            )

        if self.tfms is not None:
            transformed = self.tfms(image=image, bboxes=xyxy_bboxes, labels=class_ids)
            image = transformed["image"]
            xyxy_bboxes = np.array(transformed["bboxes"])
            class_ids = np.array(transformed["labels"])

        return image, xyxy_bboxes, class_ids, image_id, image_hw


class ConvertPredictionClassesCallback(TrainerCallback):
    def __init__(self):
        self.lookup = coco80_to_coco91_lookup()

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        predictions = batch_output["predictions"]
        coco_80_class_ids = predictions[:, -2]
        coco_91_class_ids = torch.as_tensor(
            [self.lookup[int(c)] for c in coco_80_class_ids],
            device=predictions.device,
            dtype=predictions.dtype,
        )
        # modify batch output inplace
        batch_output["predictions"][:, -2] = coco_91_class_ids


DATA_PATH = Path("/".join(Path(__file__).absolute().parts[:-2])) / "data/coco"


@script
def run_evaluation(
    coco_valid_images_path: str = DATA_PATH / "val2017",
    coco_valid_annotations_file_path: str = DATA_PATH
    / "annotations/instances_val2017.json",
    image_size: int = 640,
    architecture: str = "yolov7",
    batch_size: int = 40,
):
    ds = COCOBaseDataset(
        coco_valid_images_path,
        coco_valid_annotations_file_path,
        tfms=A.Compose(
            [
                A.LongestMaxSize(image_size),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        ),
    )

    eval_yds = Yolov7Dataset(
        ds,
        create_yolov7_transforms(training=False, image_size=(image_size, image_size)),
    )

    model = create_yolov7_model(architecture=architecture, pretrained=True)

    trainer = Yolov7Trainer(
        model=model,
        optimizer=None,
        loss_func=create_yolov7_loss(model, image_size=image_size),
        filter_eval_predictions_fn=partial(
            filter_eval_predictions, confidence_threshold=0.001
        ),
        callbacks=[
            ConvertPredictionClassesCallback,
            CalculateMeanAveragePrecisionCallback(
                targets_json=ds.targets_json, verbose=True
            ),
            *get_default_callbacks(progress_bar=True),
        ],
    )

    trainer.evaluate(
        dataset=eval_yds,
        per_device_batch_size=batch_size,
        collate_fn=yolov7_collate_fn,
    )


if __name__ == "__main__":
    run_evaluation()
