from yolov7.loss import Yolov7Loss, Yolov7LossWithAux
from yolov7.migrated.loss import ComputeLossAuxOTA, ComputeLossOTA, ComputeLoss


def create_yolov7_loss(
    model,
    image_size=640,
    box_loss_weight=0.05,
    cls_loss_weight=0.3,
    obj_loss_weight=0.7,
    aux_loss=False,
    ota_loss=True,
):
    kwargs = {
        "box_loss_weight": box_loss_weight,
        "cls_loss_weight": cls_loss_weight,
        "obj_loss_weight": obj_loss_weight,
        "image_size": image_size,
        "max_anchor_box_target_size_ratio": 4.0,

    }

    if aux_loss and ota_loss:
        loss_fn = Yolov7LossWithAux(model, **kwargs)
    elif ota_loss and not aux_loss:
        loss_fn = Yolov7Loss(model, **kwargs)
    else:
        loss_fn = Yolov7Loss(model, **kwargs)
        loss_fn.eval()

    return loss_fn


def create_yolov7_loss_orig(
    model,
    image_size=640,
    box_loss_weight=0.05,
    cls_loss_weight=0.3,
    obj_loss_weight=0.7,
    aux_loss=False,
    ota_loss=True,
):
    hyp = {}

    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    hyp["box"] = box_loss_weight * 3.0 / nl  # scale to layers
    hyp["cls"] = (
        cls_loss_weight * model.nc / 80.0 * 3.0 / nl
    )  # scale to classes and layers
    hyp["obj"] = (
        obj_loss_weight * (image_size / 640) ** 2 * 3.0 / nl
    )  # scale to image size and layers

    hyp["cls_pw"] = 1  # cls BCELoss positive_weight
    hyp["obj_pw"] = 1  # obj BCELoss positive_weight
    hyp["fl_gamma"] = 0  # focal loss gamma (efficientDet default gamma=1.5)
    hyp["anchor_t"] = 4.0  # anchor-multiple threshold

    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)

    if aux_loss and ota_loss:
        loss_fn = ComputeLossAuxOTA(model)
    elif ota_loss and not aux_loss:
        loss_fn = ComputeLossOTA(model)
    else:
        loss_fn = ComputeLoss(model)

    return loss_fn
