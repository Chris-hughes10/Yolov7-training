import torch
from tqdm import tqdm
from scipy.cluster.vq import kmeans
import numpy as np


def calculate_anchors(
    current_anchors,
    image_sizes,
    normalized_wh,
    target_image_size=1280,
    gen=1000,
    verbose=False,
    anchor_thr=4,
):
    # image sizes array of [w, h] , either np.array([[w, h]]) or per image

    # find target image sizes given longest side
    target_image_sizes = (
        target_image_size * image_sizes / image_sizes.max(1, keepdims=True)
    )

    # find wh of boxes for target size
    wh = target_image_sizes * normalized_wh

    tiny_boxes_exist = (wh < 3).any(1).sum()
    if tiny_boxes_exist:
        print(
            f"WARNING: Extremely small objects found. {tiny_boxes_exist} of {len(wh)} labels are < 3 pixels in size."
        )
        wh = wh[(wh >= 2.0).any(1)]  # filter > 2 pixels

    # bpr is achieved if each bbox is mapped to optimal anchor
    bpr, aat, _, _ = metric(current_anchors, wh, anchor_thr)

    print(f"anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}", end="")
    if bpr < 0.98:  # threshold to recompute
        print(". Attempting to improve anchors...")
        num_anchors = current_anchors.shape[0]  # number of anchors

        print(f"Running kmeans for {num_anchors} anchors on {len(wh)} points...")
        std_dev = wh.std(0)  # sigmas for whitening
        proposed_anchors, mean_dist = kmeans(
            wh / std_dev, num_anchors, iter=30
        )  # points, mean distance
        proposed_anchors *= std_dev
        wh = torch.tensor(wh, dtype=torch.float32)

        f, sh, mp, s = (
            anchor_fitness(proposed_anchors, wh, anchor_thr),
            proposed_anchors.shape,
            0.9,
            0.1,
        )  # fitness, generations, mutation prob, sigma
        pbar = tqdm(
            range(gen), desc=f"Evolving anchors with Genetic Algorithm:"
        )  # progress bar
        for _ in pbar:
            v = np.ones(sh)
            while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
                v = (
                    (np.random.random(sh) < mp)
                    * np.random.random()
                    * np.random.randn(*sh)
                    * s
                    + 1
                ).clip(0.3, 3.0)
            kg = (proposed_anchors.copy() * v).clip(min=2.0)
            fg = anchor_fitness(kg, wh, 1 / anchor_thr)
            if fg > f:
                f, proposed_anchors = fg, kg.copy()
                pbar.desc = (
                    f"Evolving anchors with Genetic Algorithm: fitness = {f:.4f}"
                )
                if verbose:
                    print_results(
                        proposed_anchors,
                        wh,
                        num_anchors,
                        1 / anchor_thr,
                        img_size=target_image_size,
                    )

        new_bpr, _, _, _ = metric(proposed_anchors, wh)
        if new_bpr > bpr:  # replace anchors
            print("New anchors better")
            current_anchors = proposed_anchors
            print_results(
                current_anchors,
                wh,
                num_anchors,
                1 / anchor_thr,
                img_size=target_image_size,
            )
        else:
            print(
                f"Original anchors better than new anchors. Proceeding with original anchors."
            )

    return current_anchors


def check_anchor_order(m):
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print("Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def update_model_anchors_inplace(model, anchors):
    detect_module = (
        model.module.model[-1] if hasattr(model, "module") else model.model[-1]
    )  # Detect()

    anchors = torch.tensor(anchors, device=detect_module.anchors.device).type_as(
        detect_module.anchors
    )
    detect_module.anchor_grid[:] = anchors.clone().view_as(
        detect_module.anchor_grid
    )  # for inference
    detect_module.anchors[:] = anchors.clone().view_as(
        detect_module.anchors
    ) / detect_module.stride.to(detect_module.anchors.device).view(
        -1, 1, 1
    )  # loss
    check_anchor_order(detect_module)


def metric(anchors, wh, anchor_thr=4):  # compute metric
    # calculates the ratio of the height and width of each bounding box to each anchor template
    r = wh[:, None] / anchors[None]
    # find min aspect ratio for each bbox
    ratio_metric = torch.min(r, 1.0 / r).min(2)[0]
    best_ratio = ratio_metric.max(1)[0]  # best_x
    aat = (
        (ratio_metric > 1.0 / anchor_thr).float().sum(1).mean()
    )  # anchors above threshold
    bpr = (best_ratio > 1.0 / anchor_thr).float().mean()  # best possible recall
    return bpr, aat, ratio_metric, best_ratio


def anchor_fitness(anchors, wh, anchor_thr=0.25):  # mutation fitness
    _, _, _, best = metric(torch.tensor(anchors, dtype=torch.float32), wh)
    return (best * (best > anchor_thr).float()).mean()  # fitness


def print_results(k, wh, n, anchor_thr=0.25, img_size=1280):
    k = k[np.argsort(k.prod(1))]  # sort small to large
    bpr, aat, x, best = metric(k, wh)
    print(
        f"anchor_thr={anchor_thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr"
    )
    print(
        f"n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, "
        f"past_thr={x[x > anchor_thr].mean():.3f}-mean: ",
        end="",
    )
    for i, x in enumerate(k):
        print(
            "%i,%i" % (round(x[0]), round(x[1])), end=",  " if i < len(k) - 1 else "\n"
        )  # use in *.cfg
    return k
