import random

import albumentations as A
import numpy as np
import torch


def _apply_transform(transform, image, boxes, classes):
    transformed = transform(image=image, bboxes=boxes, labels=classes)

    image = transformed["image"]
    boxes = np.array(transformed["bboxes"])
    classes = np.array(transformed["labels"])

    return image, boxes, classes


def create_post_mosaic_transform(
    output_height,
    output_width,
    pad_colour=(114, 114, 114),
    rotation_range=(-10, 10),
    shear_range=(-10, 10),
    translation_percent_range=(-0.2, 0.2),
    scale_range=(0.08, 1.0),
    apply_prob=0.8,
):
    return A.Compose(
        [
            A.Affine(
                cval=pad_colour,
                rotate=rotation_range,
                shear=shear_range,
                translate_percent=translation_percent_range,
                scale=None,
                keep_ratio=True,
                p=apply_prob,
            ),
            A.HorizontalFlip(),
            A.RandomResizedCrop(
                height=output_height, width=output_width, scale=scale_range
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"], min_visibility=0.25
        ),
    )


def create_pre_mixup_transform(flip_prob=0.5, **kwargs):
    return A.Compose(
        [A.HorizontalFlip(p=flip_prob)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


class MosaicBuilder:
    def __init__(self, pad_colour=(144, 144, 144)):
        self.pad_colour = pad_colour

    def create_mosaic(self, images, boxes, classes):

        mosaic_width = max(
            images[0].shape[1] + images[1].shape[1],
            images[2].shape[1] + images[3].shape[1],
        )
        mosaic_height = max(
            images[0].shape[0] + images[2].shape[0],
            images[1].shape[0] + images[3].shape[0],
        )

        mosaic_image = np.full(
            (mosaic_height, mosaic_width, 3),
            self.pad_colour,
            dtype=np.uint8,
        )

        centre_x, centre_y = self._get_mosaic_centre(mosaic_height, mosaic_width)

        mosaic_labels = []

        for mosaic_position, (image, image_boxes, image_classes) in enumerate(
            zip(images, boxes, classes)
        ):

            # concat boxes and classes for easier processing
            if len(image_boxes) > 0:
                _labels = np.concatenate((image_boxes, image_classes[None].T), axis=1)
            else:
                _labels = np.array([])

            # generate output mosaic image
            (image_height, image_width, c) = image.shape[:3]

            (mosaic_x1, mosaic_y1, mosaic_x2, mosaic_y2), (
                image_x1,
                image_y1,
                image_x2,
                image_y2,
            ) = self._get_mosaic_coordinates(
                mosaic_position,
                centre_x,
                centre_y,
                image_height,
                image_width,
                mosaic_height,
                mosaic_width,
            )

            mosaic_image[mosaic_y1:mosaic_y2, mosaic_x1:mosaic_x2] = image[
                image_y1:image_y2, image_x1:image_x2
            ]

            shift_x = mosaic_x1 - image_x1
            shift_y = mosaic_y1 - image_y1

            labels = _labels.copy()
            if labels.size > 0:
                labels = apply_shift_to_labels(labels, shift_x, shift_y)
            mosaic_labels.append(labels)

        # filter empty labels
        mosaic_labels = [labels for labels in mosaic_labels if len(labels) > 0]
        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            clip_labels_inplace(
                mosaic_labels,
                output_height=mosaic_height,
                output_width=mosaic_width,
            )

            valid_boxes = (mosaic_labels[:, 2] > mosaic_labels[:, 0]) & (
                mosaic_labels[:, 3] > mosaic_labels[:, 1]
            )
            mosaic_labels = mosaic_labels[valid_boxes]
            mosaic_boxes = mosaic_labels[:, :4]
            mosaic_classes = mosaic_labels[:, 4]

        if len(mosaic_labels) == 0:
            mosaic_boxes = np.array([])
            mosaic_classes = np.array([])

        return mosaic_image, mosaic_boxes, mosaic_classes, (mosaic_height, mosaic_width)

    def _get_mosaic_centre(self, mosaic_height, mosaic_width):
        centre_x = mosaic_width // 2
        centre_y = mosaic_height // 2
        return centre_x, centre_y

    def _get_mosaic_coordinates(
        self,
        position_idx,
        centre_x,
        centre_y,
        image_height,
        image_width,
        mosaic_height,
        mosaic_width,
    ):
        if position_idx == 0:  # top left
            mosaic_x1, mosaic_y1, mosaic_x2, mosaic_y2 = (
                max(centre_x - image_width, 0),
                max(centre_y - image_height, 0),
                centre_x,
                centre_y,
            )
            image_x1, image_y1, image_x2, image_y2 = (
                image_width - (mosaic_x2 - mosaic_x1),
                image_height - (mosaic_y2 - mosaic_y1),
                image_width,
                image_height,
            )

        elif position_idx == 1:  # top right
            mosaic_x1, mosaic_y1, mosaic_x2, mosaic_y2 = (
                centre_x,
                max(centre_y - image_height, 0),
                min(centre_x + image_width, mosaic_width),
                centre_y,
            )
            image_x1, image_y1, image_x2, image_y2 = (
                0,
                image_height - (mosaic_y2 - mosaic_y1),
                min(image_width, mosaic_x2 - mosaic_x1),
                image_height,
            )

        elif position_idx == 2:  # bottom left
            mosaic_x1, mosaic_y1, mosaic_x2, mosaic_y2 = (
                max(centre_x - image_width, 0),
                centre_y,
                centre_x,
                min(mosaic_height, centre_y + image_height),
            )
            image_x1, image_y1, image_x2, image_y2 = (
                image_width - (mosaic_x2 - mosaic_x1),
                0,
                image_width,
                min(mosaic_y2 - mosaic_y1, image_height),
            )

        elif position_idx == 3:  # bottom right
            mosaic_x1, mosaic_y1, mosaic_x2, mosaic_y2 = (
                centre_x,
                centre_y,
                min(centre_x + image_width, mosaic_width),
                min(mosaic_height, centre_y + image_height),
            )
            image_x1, image_y1, image_x2, image_y2 = (
                0,
                0,
                min(image_width, mosaic_x2 - mosaic_x1),
                min(mosaic_y2 - mosaic_y1, image_height),
            )

        else:
            raise ValueError("Incorrect index given, the accepted range is [0, 3]")

        return (mosaic_x1, mosaic_y1, mosaic_x2, mosaic_y2), (
            image_x1,
            image_y1,
            image_x2,
            image_y2,
        )


def mixup(image, boxes, classes, mixup_image, mixup_boxes, mixup_classes):
    if len(boxes) == 0 and len(mixup_boxes) == 0:
        mixed_boxes = np.array([])
        mixed_classes = np.array([])
    elif len(boxes) == 0:
        mixed_boxes = mixup_boxes
        mixed_classes = mixup_classes
    elif len(mixup_boxes) == 0:
        mixed_boxes = boxes
        mixed_classes = classes
    else:
        mixed_boxes = np.vstack((boxes, mixup_boxes))
        mixed_classes = np.hstack((classes, mixup_classes))

    mixed_image = 0.5 * image.astype(np.float32) + 0.5 * mixup_image.astype(np.float32)

    return mixed_image.astype(np.uint8), mixed_boxes, mixed_classes


class MosaicMixupDataset:
    """
    A dataset which takes an object detection dataset returning (image, boxes, classes, image_id, image_hw)
    and applies Mosaic and Mixup transforms. If both augmentations are enabled, mosaic
    is applied first, then Mixup.

    By default, this class expects the image, boxes (N, 4) and classes (N,) to be numpy arrays,
    with the boxes in (x1,y1,x2,y2) format, but this behaviour can be modified by
    overriding the `load_from_dataset` method.
    """

    def __init__(
        self,
        dataset,
        apply_mosaic_probability=1,
        apply_mixup_probability=1,
        pad_colour=(114, 114, 114),
        post_mosaic_transforms=None,
        pre_mixup_transforms=None,
    ):
        self._dataset = dataset
        self.apply_mosaic_probability = apply_mosaic_probability
        self.apply_mixup_probability = apply_mixup_probability
        self.pad_colour = pad_colour

        self.mosaic_builder = MosaicBuilder(
            pad_colour=pad_colour,
        )

        self._output_height = None
        self._output_width = None
        self.post_mosaic_transforms = post_mosaic_transforms
        self.pre_mixup_transforms = pre_mixup_transforms

    def get_output_size(self):
        return self._output_height, self._output_width

    def load_from_dataset(self, index):
        image, xyxy_boxes, classes, image_id, image_hw = self._dataset[index]
        return np.array(image), xyxy_boxes, classes, image_id, image_hw

    def __len__(self):
        return len(self._dataset)

    def enable(self, apply_mosaic_probability=1, apply_mixup_probability=1):
        self.apply_mosaic_probability = apply_mosaic_probability
        self.apply_mixup_probability = apply_mixup_probability

    def disable(self):
        self.apply_mosaic_probability = 0
        self.apply_mixup_probability = 0

    def __getitem__(self, index):
        apply_mosaic = random.random() <= self.apply_mosaic_probability
        if apply_mosaic:

            indices = [index] + torch.randint(
                low=0, high=len(self._dataset), size=(3,)
            ).tolist()
            random.shuffle(indices)

            mosaic_images, mosaic_boxes, mosaic_classes, image_id, orig_images_hw = zip(
                *[self.load_from_dataset(ds_index) for ds_index in indices]
            )

            image, boxes, classes, image_hw = self.mosaic_builder.create_mosaic(
                mosaic_images, mosaic_boxes, mosaic_classes
            )

            if self.post_mosaic_transforms is not None:
                image, boxes, classes = _apply_transform(
                    self.post_mosaic_transforms,
                    image=image,
                    boxes=boxes,
                    classes=classes,
                )
        else:
            image, boxes, classes, image_id, image_hw = self.load_from_dataset(index)

        if random.random() <= self.apply_mixup_probability:
            image_id = [image_id]
            (image, boxes, classes), mixup_other_image_id = self.apply_mixup(
                image, boxes, classes
            )
            image_id.append(mixup_other_image_id)

        return image, boxes, classes, image_id, image_hw

    def apply_mixup(self, image, boxes, classes):
        mixup_image = None
        mixup_boxes = []
        mixup_classes = None

        while len(mixup_boxes) == 0:
            # select a random image with labels from the dataset to use as mixup image
            mixup_image_index = random.randint(0, self.__len__() - 1)
            (
                mixup_image,
                mixup_boxes,
                mixup_classes,
                image_id,
                image_hw,
            ) = self.load_from_dataset(mixup_image_index)
            mixup_image, mixup_boxes, mixup_classes = _apply_transform(
                self.create_resize_transform(
                    output_height=image.shape[0],
                    output_width=image.shape[1],
                    pad_colour=self.pad_colour,
                ),
                image=mixup_image,
                boxes=mixup_boxes,
                classes=mixup_classes,
            )

        if self.pre_mixup_transforms is not None:
            mixup_image, mixup_boxes, mixup_classes = _apply_transform(
                self.pre_mixup_transforms, mixup_image, mixup_boxes, mixup_classes
            )

        return (
            mixup(image, boxes, classes, mixup_image, mixup_boxes, mixup_classes),
            image_id,
        )

    @staticmethod
    def create_resize_transform(output_height, output_width, pad_colour):
        return A.Compose(
            [
                A.LongestMaxSize(max(output_height, output_width)),
                A.PadIfNeeded(
                    output_height,
                    output_width,
                    border_mode=0,
                    value=pad_colour,
                ),
                A.Resize(height=output_height, width=output_width),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )


def apply_shift_to_labels(labels, shift_x, shift_y):
    labels_out = labels.copy()
    labels_out[:, 0] = labels[:, 0] + shift_x
    labels_out[:, 1] = labels[:, 1] + shift_y
    labels_out[:, 2] = labels[:, 2] + shift_x
    labels_out[:, 3] = labels[:, 3] + shift_y
    return labels_out


def clip_labels_inplace(labels, output_height, output_width):
    np.clip(labels[:, 0], 0, output_width, out=labels[:, 0])
    np.clip(labels[:, 1], 0, output_height, out=labels[:, 1])
    np.clip(labels[:, 2], 0, output_width, out=labels[:, 2])
    np.clip(labels[:, 3], 0, output_height, out=labels[:, 3])
