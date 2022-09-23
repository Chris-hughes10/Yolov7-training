import torch

PAD_VALUE = -1

def remove_padding(padded_tensor, pad_value):

    padding_mask = padded_tensor == pad_value

    if padding_mask.ndim > 1:
        padding_mask = torch.all(padding_mask, dim=-1)

    result =  padded_tensor[~padding_mask]

    return result


def test_can_remove_padding_1d():

    t = torch.tensor([0, 1, 2, 3, PAD_VALUE, PAD_VALUE])
    expected_t = torch.tensor([0, 1, 2, 3])

    actual = remove_padding(t, PAD_VALUE)

    assert torch.eq(expected_t, actual).all()

def test_can_remove_padding_2d():
    t = torch.tensor([[0, 1],
                     [1, 2],
                     [PAD_VALUE, PAD_VALUE],
                     [PAD_VALUE, PAD_VALUE]])

    expected_t = torch.tensor([[0, 1],
                      [1, 2]])

    actual = remove_padding(t, PAD_VALUE)

    assert torch.eq(expected_t, actual).all()

def test_can_remove_padding_3d():
    t = torch.tensor([[[0, 1],
                     [1, 2]],
                     [[PAD_VALUE, PAD_VALUE],
                     [PAD_VALUE, PAD_VALUE]]])

    print(t.shape)

    expected_t = torch.tensor([[[0, 1],
                      [1, 2]]])

    actual = remove_padding(t, PAD_VALUE)

    assert torch.eq(expected_t, actual).all()

