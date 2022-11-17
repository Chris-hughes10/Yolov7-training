import albumentations as A
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


def yolov7_collate_fn(batch):
    images, labels, indices, image_sizes = zip(*batch)
    for i, l in enumerate(labels):
        l[:, 0] = i  # add target image index for build_targets() in loss fn
    return (
        torch.stack(images, 0),
        torch.cat(labels, 0),
        torch.stack(indices, 0),
        torch.stack(image_sizes, 0),
    )


def create_base_transforms(target_image_size):
    return A.Compose(
        [
            A.LongestMaxSize(target_image_size),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def create_yolov7_transforms(
    image_size=(640, 640),
    training=False,
    training_transforms=(A.HorizontalFlip(p=0.5),),
):
    transforms = [
        A.LongestMaxSize(max(image_size)),
        A.PadIfNeeded(
            image_size[0],
            image_size[1],
            border_mode=0,
            value=(114, 114, 114),
        ),
    ]

    if training:
        transforms.extend(training_transforms)

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def convert_xyxy_to_cxcywh(bboxes):
    bboxes = bboxes.copy()
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


class Yolov7Dataset(Dataset):
    """
    A dataset which takes an object detection dataset returning (image, boxes, classes, image_id, image_hw)
    and applies the necessary preprocessing steps as required by Yolov7 models.

    By default, this class expects the image, boxes (N, 4) and classes (N,) to be numpy arrays,
    with the boxes in (x1,y1,x2,y2) format, but this behaviour can be modified by
    overriding the `load_from_dataset` method.
    """

    def __init__(self, dataset, transforms=None):
        self.ds = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def load_from_dataset(self, index):
        image, boxes, classes, image_id, shape = self.ds[index]
        return image, boxes, classes, image_id, shape

    def __getitem__(self, index):
        image, boxes, classes, image_id, original_image_size = self.load_from_dataset(
            index
        )

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes, labels=classes)
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"])
            classes = np.array(transformed["labels"])

        image = image / 255  # 0 - 1 range

        if len(boxes) != 0:
            # filter boxes with 0 area in any dimension
            valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid_boxes]
            classes = classes[valid_boxes]

            boxes = torchvision.ops.box_convert(
                torch.as_tensor(boxes, dtype=torch.float32), "xyxy", "cxcywh"
            )
            boxes[:, [1, 3]] /= image.shape[0]  # normalized height 0-1
            boxes[:, [0, 2]] /= image.shape[1]  # normalized width 0-1
            classes = np.expand_dims(classes, 1)

            labels_out = torch.hstack(
                (
                    torch.zeros((len(boxes), 1)),
                    torch.as_tensor(classes, dtype=torch.float32),
                    boxes,
                )
            )
        else:
            labels_out = torch.zeros((0, 6))

        try:
            if len(image_id) > 0:
                image_id_tensor = torch.as_tensor([])

        except TypeError:
            image_id_tensor = torch.as_tensor(image_id)

        return (
            torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32),
            labels_out,
            image_id_tensor,
            torch.as_tensor(original_image_size),
        )
