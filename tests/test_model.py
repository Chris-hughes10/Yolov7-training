from pathlib import Path
from yolov7.dataset import Yolov7Dataset
from yolov7.models.model_factory import create_yolov7_model
import torch
from example.train_cars import DatasetAdaptor, load_cars_df
from yolov7.dataset import create_yolov7_transforms
import pytest
import pickle
import pytorch_accelerated
from accelerate.utils import set_seed


@pytest.fixture(scope="module")
def image_tensor():
    data_path = "./cars_dataset"
    data_path = Path(data_path)
    images_path = data_path / "training_images"
    annotations_file_path = data_path / "annotations.csv"
    target_image_size = 640

    train_df, valid_df, lookups = load_cars_df(annotations_file_path, images_path)
    ds = DatasetAdaptor(images_path, train_df)
    yolo_ds = Yolov7Dataset(ds, transforms=create_yolov7_transforms(image_size=(target_image_size, target_image_size)))
    image_tensor, *_ = yolo_ds[0]
    return image_tensor


def test_model_outputs(image_tensor):
    torch.manual_seed(0)
    new_model = create_yolov7_model(architecture="yolov7-e6e", pretrained=True)
    torch.manual_seed(0)
    old_model = create_yolov7_model(architecture="yolov7-e6e", pretrained=True, legacy=True)

    # new_loss_func = create_yolov7_loss(new_model, ota_loss=True, aux_loss=False)
    # old_loss_func = create_yolov7_loss(old_model, ota_loss=True, aux_loss=False)

    new_image_tensor = image_tensor.detach().clone().requires_grad_()
    old_image_tensor = image_tensor.detach().clone().requires_grad_()
    new_outputs = new_model(new_image_tensor[None])
    old_outputs = old_model(old_image_tensor[None])

    # new_loss_func.backwards()
    # old_loss_func.backwards()

    for i, (new_o, old_o) in enumerate(zip(new_outputs, old_outputs)):
        assert (new_o.round(decimals=5) == old_o.round(decimals=5)).all()

    # assert (new_image_tensor.grad.round(decimals=5) == old_image_tensor.grad.round(decimals=5)).all()

@pytest.mark.parametrize("arch", ["yolov7", "yolov7-e6e"])
def test_outputs_are_constant(image_tensor, arch):
    set_seed(42)

    model = create_yolov7_model(architecture=arch, pretrained=True)
    with open(f"./tests/prev_outputs_{arch}.pkl", "rb+") as f:
        prev_outputs = pickle.load(f)

    outputs = model(image_tensor[None])

    for (o, prev_o) in zip(outputs, prev_outputs):
        assert (o.round(decimals=15) == prev_o.round(decimals=15)).all()




# def test_model_preds(image_tensor):
#     new_model = create_yolov7_model(architecture="yolov7", pretrained=True)
#     old_model = create_yolov7_model(architecture="yolov7", pretrained=True, legacy=True)

#     new_model.eval()
#     old_model.eval()

#     with torch.no_grad():
#         new_outputs = new_model(image_tensor[None])
#         new_preds = new_model.postprocess(new_outputs)

#         old_outputs = old_model(image_tensor[None])
#         old_preds = legacy_process_yolov7_outputs(old_outputs)

#     for i, (new_o, old_o) in enumerate(zip(new_outputs, old_outputs[1])):
#         assert (new_o.round(decimals=15) == old_o.round(decimals=15)).all()

#     assert (new_preds[0].round(decimals=15) == old_preds[0].round(decimals=15)).all()