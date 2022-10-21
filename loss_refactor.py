from pathlib import Path
import pickle

from example.data import load_cars_df, DatasetAdaptor
from yolov7 import create_yolov7_model
from yolov7.dataset import Yolov7Dataset, create_yolov7_transforms, yolov7_collate_fn

from torch.utils.data import DataLoader
import torch

from yolov7.loss_factory import create_yolov7_loss, create_yolov7_loss_orig
# Ensures seeds are set to remove randomness
import pytorch_accelerated

def main():
    data_path = "./cars_dataset"
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

    dl = DataLoader(eval_yds, collate_fn=yolov7_collate_fn, batch_size=2, shuffle=False)

    model = create_yolov7_model(
        architecture="yolov7-e6e", num_classes=2, pretrained=True
    )
    # We use trai mode for this because we need auxiliary head for aux loss

    ota_loss = True
    aux_loss = True

    eval_loss_func = create_yolov7_loss_orig(model, ota_loss=ota_loss, aux_loss=aux_loss)
    eval_loss_func_r = create_yolov7_loss(model, ota_loss=ota_loss, aux_loss=aux_loss)
    # eval_loss_func = create_yolov7_loss_orig(model, ota_loss=True, aux_loss=True)
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
            loss_inputs = {"model_outputs": model_outputs, "labels": labels, "images": images}
            # Uncomment to save variables to use in testing
            # with open(f"batch{i}_loss_inputs.pkl", "wb") as f:
            #     pickle.dump(loss_inputs, f)
            if not aux_loss:
                # Only need auxiliary head outputs for aux loss
                # eval mode prunes the same way because it only uses no aux loss (nor OTA but no prob)
                model_outputs = model_outputs[:model.model[-1].nl]

            # inference_outputs, rpn_outputs = model_outputs
            val_loss, loss_items = eval_loss_func(p=model_outputs, targets=labels, imgs=images)
            val_loss_r, loss_items_r = eval_loss_func_r(p=model_outputs, targets=labels, imgs=images)

            assert val_loss_r == val_loss
            assert (loss_items_r == loss_items).all()

            i +=1

            if i == 3:
                break

    print('Done')

if __name__ == '__main__':
    main()