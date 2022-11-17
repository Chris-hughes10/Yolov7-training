from yolov7.loss import Yolov7Loss, Yolov7LossWithAux

# from yolov7.migrated.loss import ComputeLossAuxOTA, ComputeLossOTA, ComputeLoss


def create_yolov7_loss(
    model,
    image_size=640,
    box_loss_weight=0.05,
    cls_loss_weight=0.3,
    obj_loss_weight=0.7,
    ota_loss=True,
):
    kwargs = {
        "box_loss_weight": box_loss_weight,
        "cls_loss_weight": cls_loss_weight,
        "obj_loss_weight": obj_loss_weight,
        "image_size": image_size,
        "max_anchor_box_target_size_ratio": 4.0,
    }

    use_aux_loss = model.config["aux_detection"]

    if use_aux_loss and ota_loss:
        loss_fn = Yolov7LossWithAux(model, **kwargs)
    elif ota_loss and not use_aux_loss:
        loss_fn = Yolov7Loss(model, **kwargs)
    else:
        loss_fn = Yolov7Loss(model, **kwargs)
        loss_fn.eval()

    return loss_fn
