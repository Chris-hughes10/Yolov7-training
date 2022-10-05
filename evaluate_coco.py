import os

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_accelerated import notebook_launcher
from pytorch_accelerated.callbacks import get_default_callbacks
from torchvision.datasets.coco import CocoDetection

from yolov7 import create_yolov7_model
from yolov7.dataset import Yolov7Dataset, create_yolov7_transforms, yolov7_collate_fn
from yolov7.evaluation import CalculateMetricsCallback
from yolov7.evaluation import CalculateMetricsCallbackV2
from yolov7.loss_factory import create_yolov7_loss
from yolov7.trainer import Yolov7Trainer


def coco91_to_coco80_lookup():
    return {
        v: i
        for i, v in enumerate(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                27,
                28,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                67,
                70,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
            ]
        )
    }


class COCOBaseDataset(CocoDetection):
    def __init__(self, img_dir, annotation_path):
        super().__init__(root=str(img_dir), annFile=str(annotation_path))
        self.lookup = coco91_to_coco80_lookup()
        # image_size = (640, 640)
        image_size = (736, 736)
        self.a_transforms = A.Compose(
            [A.LongestMaxSize(max(image_size)),
            # A.PadIfNeeded(
            #     image_size[0],
            #     image_size[1],
            #     border_mode=0,
            #     value=(114, 114, 114),
            # ),
        # A.Resize(640, 640)
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        img = np.array(Image.open(os.path.join(self.root, path)).convert("RGB"))

        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        # r = 640 / max(h0, w0)  # resize image to img_size
        # if r != 1:  # always resize down, only resize up if training with augmentation
        #     interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        #     img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0)

    def __getitem__(self, index):
        image_id = self.ids[index]

        # if image_id == 7386:
        #     print('break')
        image_tup, targets = super().__getitem__(index)
        image, shape = image_tup
        # image = np.array(image)
        raw_boxes = [target["bbox"] for target in targets if target['iscrowd'] == 0]


        if len(raw_boxes) == 0:
            xyxy_bboxes = np.array([])
            class_ids = np.array([])
        else:
            xyxy_bboxes = torchvision.ops.box_convert(
                torch.as_tensor(raw_boxes), "xywh", "xyxy"
            ).numpy()
            class_ids = np.array(
                [self.lookup[target["category_id"]] for target in targets if target['iscrowd'] == 0]
            )
            # class_ids = np.array(
            #     [target["category_id"] for target in targets if target['iscrowd'] == 0]
            # )
        if self.a_transforms is not None:
            transformed = self.a_transforms(image=image, bboxes=xyxy_bboxes, labels=class_ids)
            image = transformed["image"]
            xyxy_bboxes = np.array(transformed["bboxes"])
            class_ids = np.array(transformed["labels"])

        return image, xyxy_bboxes, class_ids, image_id, shape


def main():

    ds = COCOBaseDataset(
        "/home/chris/notebooks/Yolov7-training/coco_dataset/coco/images/val2017",
        "/home/chris/notebooks/Yolov7-training/coco_dataset/coco/annotations/instances_val2017.json",
    )

    eval_yds = Yolov7Dataset(
        ds, create_yolov7_transforms(training=False, image_size=(736, 736))
        # ds, create_yolov7_transforms(training=False, image_size=(640, 640))
    )

    model = create_yolov7_model(architecture="yolov7", pretrained=True, training=True)

    trainer = Yolov7Trainer(
        model=model,
        optimizer=None,
        loss_func=create_yolov7_loss(model, ota_loss=False),
        eval_loss_func=create_yolov7_loss(model, ota_loss=False),
        callbacks=[
            CalculateMetricsCallbackV2(),
            *get_default_callbacks(progress_bar=True),
        ],
    )

    trainer.evaluate(
        dataset=eval_yds,
        per_device_batch_size=40,
        # dataloader_kwargs={'num_workers': 0},
        collate_fn=yolov7_collate_fn,
    )

    print("done")


if __name__ == "__main__":
    os.environ["mixed_precision"] = "fp16"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    notebook_launcher(main, num_processes=2)
    # main()
