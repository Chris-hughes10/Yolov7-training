"""Tests to use for refactoring the loss. Inputs and outputs from cars ds run."""
import pickle
import torch
from yolov7.loss_factory import create_yolov7_loss

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
"""
Loss values
0, core, 3.98261, tensor([0.04894, 1.93461, 0.00775, 1.99130])
1, core, 3.98727, tensor([0.05133, 1.93455, 0.00776, 1.99363])
2, core, 3.99030, tensor([0.05282, 1.93455, 0.00777, 1.99515])

0, ota, 3.94623, tensor([0.03071, 1.93461, 0.00779, 1.97312])
1, ota, 3.94977, tensor([0.03254, 1.93455, 0.00780, 1.97489])
2, ota, 3.96024, tensor([0.03778, 1.93456, 0.00778, 1.98012])

0, ota aux, 5.09125, tensor([0.04916, 2.48746, 0.00900, 2.54563])
1, ota aux, 5.09861, tensor([0.05321, 2.48767, 0.00843, 2.54930])
2, ota aux, 5.10369, tensor([0.05612, 2.48728, 0.00844, 2.55184])
"""

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

def test_loss_batch_1():
    model = FakeModel()
    loss_func = create_yolov7_loss(model, ota_loss=True, aux_loss=False)

    with open("./tests/batch1inputs.pkl", "rb") as f:
        inputs = pickle.load(f)
    rpn_outputs = inputs["rpn_outputs"]
    labels = inputs["labels"]
    images = inputs["images"]

    expected_loss = torch.tensor([5.40037])
    expected_loss_items = torch.tensor([0.07192, 2.62826, 0.00000, 2.70018])

    loss, loss_items = loss_func(p=rpn_outputs, targets=labels, imgs=images)

    assert (loss.round(decimals=5) == expected_loss.round(decimals=5)).all().item()
    assert (loss_items.round(decimals=5) == expected_loss_items.round(decimals=5)).all().item()