from pathlib import Path

from example.data import load_cars_df, DatasetAdaptor
from yolov7 import create_yolov7_model
from yolov7.dataset import Yolov7Dataset, create_yolov7_transforms, yolov7_collate_fn

from torch.utils.data import DataLoader
import torch

from yolov7.loss_factory import create_yolov7_loss, create_yolov7_loss_orig


def main():
    data_path = "/home/chris/Downloads/data"
    # data_path = r"C:\Users\hughesc\Documents\data\cars\data"
    data_path = Path(data_path)
    images_path = data_path / "training_images"
    annotations_file_path = data_path / "annotations.csv"

    train_df, valid_df, lookups = load_cars_df(annotations_file_path, images_path)

    label_to_class_id = {"car": 0}

    eval_ds = DatasetAdaptor(images_path, valid_df, label_to_class_id, bgr_images=False)

    eval_yds = Yolov7Dataset(
        eval_ds, create_yolov7_transforms(training=False, image_size=(640, 640))
    )

    dl = DataLoader(eval_yds, collate_fn=yolov7_collate_fn, batch_size=2)

    model = create_yolov7_model(
        architecture="yolov7", num_classes=1, pretrained=True
    )

    model.eval()

    eval_loss_func_r = create_yolov7_loss(model, ota_loss=True, aux_loss=False)
    eval_loss_func = create_yolov7_loss_orig(model, ota_loss=True, aux_loss=False)
    i = 0

    for batch in dl:
        with torch.no_grad():
            images, labels, image_idxs, original_image_sizes = (
                batch[0],
                batch[1],
                batch[2],
                batch[3].cpu(),
            )

            model_outputs = model(images)

            inference_outputs, rpn_outputs = model_outputs
            val_loss_r, loss_items_r = eval_loss_func_r(p=rpn_outputs, targets=labels, imgs=images)
            val_loss, loss_items = eval_loss_func(p=rpn_outputs, targets=labels, imgs=images)

            assert val_loss_r == val_loss
            assert (loss_items_r == loss_items).all()

            i +=1

            if i == 10:
                break

    print('Done')

if __name__ == '__main__':
    main()