"""Tests to use for refactoring the loss. Inputs and outputs from cars ds run."""
import pickle

import pytest
import torch

from yolov7.loss_factory import create_yolov7_loss, create_yolov7_loss_orig



class FakeDetector:
    """Class that has needed attributes by the loss to work but is not the real thing

    Obtained from the yolov7-e6e model with 2 classes
    """
    nl = 4
    na = 3
    nc = 2
    stride = torch.tensor([8., 16., 32., 64.])
    anchors = torch.tensor([[[2.375, 3.375],
                             [5.5, 5.0],
                             [4.75, 11.75]],
                            [[6.0, 4.25],
                             [5.375, 9.5],
                             [11.25, 8.5625]],
                            [[4.375, 9.40625],
                             [9.46875, 8.25],
                             [7.4375, 16.9375]],
                            [[6.8125, 9.60938],
                             [11.54688, 5.9375],
                             [14.45312, 12.375]]])


class FakeModel:
    """Class that has the attributes needed by the loss to work but is not really a model

    Obtained from the yolov7-e6e model with 2 classes
    """
    model = [FakeDetector()]
    nc = 2

    def parameters(self):
        class Dummy:
            device = torch.device("cpu")
        return iter([Dummy()])


@pytest.fixture(scope="module")
def batch_loss_inputs():
    "Load pkl files only once per execution"
    result = {}
    for i in range(3):
        with open(f"./tests/batch{i}_loss_inputs.pkl", "rb") as f:
            result[i] = pickle.load(f)
    return result

# Loss values were obtained by running e6e model through first 2-sample 3 batches of cars dataset
@pytest.mark.parametrize(
    "batch, ota_loss, aux_loss, expected_loss, expected_loss_items",
    [
        # core loss
        (0, False, False, 3.98261, [0.04894, 1.93461, 0.00775, 1.99130]),
        (1, False, False, 3.98727, [0.05133, 1.93455, 0.00776, 1.99363]),
        (2, False, False, 3.99030, [0.05282, 1.93455, 0.00777, 1.99515]),
        # ota loss
        (0, True, False, 3.94623, [0.03071, 1.93461, 0.00779, 1.97312]),
        (1, True, False, 3.94977, [0.03254, 1.93455, 0.00780, 1.97489]),
        (2, True, False, 3.96024, [0.03778, 1.93456, 0.00778, 1.98012]),
        # aux ota loss
        (0, True, True, 5.09125, [0.04916, 2.48746, 0.00900, 2.54562]),
        (1, True, True, 5.09861, [0.05321, 2.48767, 0.00843, 2.54930]),
        (2, True, True, 5.10369, [0.05612, 2.48728, 0.00844, 2.55184]),
    ]
)
def test_loss(batch, ota_loss, aux_loss, expected_loss, expected_loss_items, batch_loss_inputs):
    model = FakeModel()
    loss_func = create_yolov7_loss(model, ota_loss=ota_loss, aux_loss=aux_loss)
    prev_loss_func = create_yolov7_loss_orig(model, ota_loss=ota_loss, aux_loss=aux_loss)
    expected_loss = torch.tensor([expected_loss])
    expected_loss_items = torch.tensor(expected_loss_items)


    model_outputs = [o.detach().clone().requires_grad_() for o in batch_loss_inputs[batch]["model_outputs"]]
    labels = batch_loss_inputs[batch]["labels"]
    images = batch_loss_inputs[batch]["images"]
    if not aux_loss:
        # Only the aux loss requires the aux head outputs, for the rest we strip them
         model_outputs = model_outputs[:model.model[-1].nl]
    prev_model_outputs = [o.detach().clone().requires_grad_() for o in model_outputs]

    loss, loss_items = loss_func(model_outputs, targets=labels, images=images)
    prev_loss, prev_loss_items = prev_loss_func(p=prev_model_outputs, targets=labels, imgs=images)

    assert (loss.round(decimals=5) == expected_loss.round(decimals=5)).all().item()
    assert (loss_items.round(decimals=5) == expected_loss_items.round(decimals=5)).all().item()

    loss.backward()
    prev_loss.backward()
    for i in range(len(model_outputs)):
        assert (model_outputs[i].grad == prev_model_outputs[i].grad).all()
