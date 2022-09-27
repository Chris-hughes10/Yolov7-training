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
    get_rectangle_corners_fn=get_rectangle_params_from_pascal_bbox,
):
    for bbox in bboxes:
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=4,
            edgecolor="black",
            fill=False,
        )
        rect_2 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=2,
            edgecolor="white",
            fill=False,
        )

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)


draw_bboxes_coco = partial(
    draw_bboxes, get_rectangle_corners_fn=get_rectangle_params_from_coco_bbox
)

draw_bboxes_yolo = partial(
    draw_bboxes, get_rectangle_corners_fn=get_rectangle_params_from_yolo_bbox
)

draw_bboxes_pascal_voc = partial(
    draw_bboxes, get_rectangle_corners_fn=get_rectangle_params_from_pascal_bbox
)

draw_functions = {
    "coco": draw_bboxes_coco,
    "yolo": draw_bboxes_yolo,
    "pascal": draw_bboxes_pascal_voc,
}


def annotate_image(image, bboxes=None, bbox_format="pascal"):
    draw_bboxes_fn = draw_functions[bbox_format]

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    if bboxes:
        draw_bboxes_fn(ax, bboxes)

    return fig


def show_image(image, bboxes=None, bbox_format="pascal"):
    fig = annotate_image(image, bboxes, bbox_format)
    plt.show()
