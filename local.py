from pathlib import Path
import torch
from yolov7.models.model_factory import create_yolov7_model
from yolov7.dataset import Yolov7Dataset
from example.cars_data_utils import DatasetAdaptor, load_cars_df
from yolov7.dataset import create_yolov7_transforms


if __name__ == "__main__":
    data_path = "./cars_dataset"
    data_path = Path(data_path)
    images_path = data_path / "training_images"
    annotations_file_path = data_path / "annotations.csv"
    target_image_size = 640

    train_df, valid_df, lookups = load_cars_df(annotations_file_path, images_path)
    ds = DatasetAdaptor(images_path, train_df)
    yolo_ds = Yolov7Dataset(
        ds,
        transforms=create_yolov7_transforms(
            image_size=(target_image_size, target_image_size)
        ),
    )
    image_tensor, labels, image_id, image_size = yolo_ds[0]

    model = create_yolov7_model(architecture="yolov7", num_classes=2, pretrained=True)
    with torch.no_grad():
        model_outputs = model(image_tensor[None])
        preds = model.postprocess(model_outputs)
