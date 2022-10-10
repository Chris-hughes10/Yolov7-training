from functools import partial

import matplotlib.pyplot as plt
from matplotlib import patches


def get_rectangle_params_from_coco_bbox(bbox):
    x_min, y_min, width, height = bbox

    bottom_left = (x_min, y_min)

    return bottom_left, width, height


def get_rectangle_params_from_yolo_bbox(bbox):
    cx, cy, width, height = bbox

    x_min = cx - width / 2
    y_min = cy - height / 2

    bottom_left = (x_min, y_min)

    return bottom_left, width, height


def get_rectangle_params_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height


def draw_bboxes(
        plot_ax,
        bboxes,
        class_labels=None,
        color_1="black",
        color_2="white",
        get_rectangle_corners_fn=get_rectangle_params_from_pascal_bbox,
):
    if class_labels is not None:
        assert len(class_labels) == len(bboxes)
    else:
        class_labels = [None] * len(bboxes)

    for bbox, label in zip(bboxes, class_labels):
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=4,
            edgecolor=color_1,
            fill=False,
        )
        rect_2 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=2,
            edgecolor=color_2,
            fill=False,
        )

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)

        if label is not None:
            rx, ry = rect_1.get_xy()
            plot_ax.annotate(label, (rx, ry + height), color=color_2, fontsize=20)


draw_bboxes_coco = partial(
    draw_bboxes, get_rectangle_corners_fn=get_rectangle_params_from_coco_bbox
)

draw_bboxes_yolo = partial(
    draw_bboxes, get_rectangle_corners_fn=get_rectangle_params_from_yolo_bbox
)

draw_xyxy_bboxes = partial(
    draw_bboxes, get_rectangle_corners_fn=get_rectangle_params_from_pascal_bbox
)

draw_functions = {
    "coco": draw_bboxes_coco,
    "cxcywh": draw_bboxes_yolo,
    "xyxy": draw_xyxy_bboxes,
}


def annotate_image(image, bboxes=None, class_labels=None, bbox_format="xyxy", close_fig=True):
    draw_bboxes_fn = draw_functions[bbox_format]

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    if bboxes:
        draw_bboxes_fn(ax, bboxes, class_labels)

    if close_fig:
        plt.close(fig)

    return fig


def show_image(image, bboxes=None, class_labels=None, bbox_format="xyxy"):
    fig = annotate_image(image, bboxes, class_labels, bbox_format, close_fig=False)
    plt.show()
