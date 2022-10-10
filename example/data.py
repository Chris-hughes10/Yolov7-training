import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


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

    class_id_to_label = dict(enumerate(df.query('has_annotation == True').class_name.unique()))
    class_label_to_id = {v: k for k, v in class_id_to_label.items()}

    df["image_id"] = df.image.map(image_to_image_id)
    df["class_id"] = df.class_name.map(class_id_to_label)

    file_names = tuple(df.image.unique())
    random.seed(42)
    validation_files = set(random.sample(file_names, int(len(df) * 0.2)))
    train_df = df[~df.image.isin(validation_files)]
    valid_df = df[df.image.isin(validation_files)]

    lookups = {
        "image_id_to_image": image_id_to_image,
        "image_to_image_id": image_to_image_id,
        "class_id_to_label": class_id_to_label,
        "class_label_to_id": class_label_to_id
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
