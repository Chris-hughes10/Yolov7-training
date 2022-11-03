<<<<<<< HEAD
import numpy as np
import torch
from scipy.cluster.vq import kmeans
from tqdm import tqdm

# anchor width and height multiple threshold used to select label-anchor matches when computing loss
LOSS_ANCHOR_MULTIPLE_THRESHOLD = 4


def calculate_resized_gt_wh(gt_wh, image_sizes, target_image_size=640):
    """
    Given an array of bounding box  widths and heights, and their corresponding image sizes,
    resize these relative to the specified target image size.

    This function assumes that resizing will be performed by scaling the image such that the longest
    side is equal to the given target image size.

    :param gt_wh: an array of shape [N, 2] containing the raw width and height of each box.
    :param image_sizes: an array of shape [N, 2] or [1, 2] containing the width and height of the image corresponding to each box.
    :param target_image_size: the size of the images that will be used during training.

    """
    normalized_gt_wh = gt_wh / image_sizes
    target_image_sizes = (
        target_image_size * image_sizes / image_sizes.max(1, keepdims=True)
    )

    resized_gt_wh = target_image_sizes * normalized_gt_wh

    tiny_boxes_exist = (resized_gt_wh < 3).any(1).sum()
    if tiny_boxes_exist:
        print(
            f"""WARNING: Extremely small objects found. 
            {tiny_boxes_exist} of {len(resized_gt_wh)} labels are < 3 pixels in size. These will be removed
            """
        )
        resized_gt_wh = resized_gt_wh[(resized_gt_wh >= 2.0).any(1)]

    return resized_gt_wh


def calculate_best_possible_recall(anchors, gt_wh):
    """
    Given a tensor of anchors and and an array of widths and heights for each bounding box in the dataset,
    calculate the best possible recall that can be obtained if every box was matched to an appropriate anchor.

    :param anchors: a tensor of shape [N, 2] representing the width and height of each anchor
    :param gt_wh: a tensor of shape [N, 2] representing the width and height of each ground truth bounding box

    """
    best_anchor_ratio = calculate_best_anchor_ratio(anchors=anchors, gt_wh=gt_wh)
    best_possible_recall = (
        (best_anchor_ratio > 1.0 / LOSS_ANCHOR_MULTIPLE_THRESHOLD).float().mean()
    )

    return best_possible_recall


def calculate_best_anchor_ratio(anchors, gt_wh):
    """
    Given a tensor of anchors and and an array of widths and heights for each bounding box in the dataset,
    calculate the ratios indicaticating of how well the most suitable anchor box matches each ground truth target.
    These values will be in the range [0, 1].

    This is done by calculating the ratios of its height and width when compared to the height and width of the
    ground truth targets, as an indication of how well, independently, the width and height of each anchor box matches
    each ground truth target.

    Taking the minimum ratio for each anchor, which represents the side that worst matches the ground truth,
    the anchor with the highest score is selected as the matched anchor and this ratio is returned.

    :param anchors: a tensor of shape [N, 2] representing the width and height of each anchor
    :param gt_wh: a tensor of shape [N, 2] representing the width and height of each ground truth bounding box

    """
    gt_wh = torch.as_tensor(gt_wh, dtype=torch.float32)
    symmetric_size_ratios = torch.min(
        gt_wh[:, None] / anchors[None], anchors[None] / gt_wh[:, None]
    )  # ensure 0-1 range
    worst_side_size_ratio = symmetric_size_ratios.min(-1).values
    best_anchor_ratio = worst_side_size_ratio.max(-1).values
    return best_anchor_ratio


def calculate_anchors(
    current_anchors,
    image_sizes,
    gt_wh,
    target_image_size=640,
    best_possible_recall_threshold=0.98,
):
    # image sizes array of [w, h] , either np.array([[w, h]]) or per image
    gt_wh = calculate_resized_gt_wh(
        image_sizes=image_sizes,
        gt_wh=gt_wh,
        target_image_size=target_image_size,
    )

    best_possible_recall = calculate_best_possible_recall(
        anchors=current_anchors, gt_wh=gt_wh
    )

    print(
        f"Best Possible Recall (BPR) = {best_possible_recall:.4f}",
        end="",
    )
    if best_possible_recall < best_possible_recall_threshold:
        print(". Attempting to improve anchors...")
        num_anchors = current_anchors.shape[0]
        proposed_anchors = estimate_anchors(num_anchors=num_anchors, gt_wh=gt_wh)
        proposed_anchors = evolve_anchors(
            proposed_anchors=proposed_anchors, gt_wh=gt_wh
        )
        proposed_anchors = sort_anchors(proposed_anchors)

        new_best_possible_recall = calculate_best_possible_recall(
            proposed_anchors, gt_wh=gt_wh
        )

        print(
            f"Best Possible Recall of new anchors = {best_possible_recall:.4f}",
            end="\n",
        )

        if new_best_possible_recall > best_possible_recall:
            print(
                f"Best Possible Recall of new anchors is better than original anchors. Proceeding with new anchors",
                end="",
            )
            current_anchors = proposed_anchors
        else:
            print(
                f"Original anchors equal or better than new anchors. Proceeding with original anchors."
            )

    return current_anchors


def estimate_anchors(num_anchors, gt_wh):
    """
    Given a target number of anchors and an array of widths and heights for each bounding box in the dataset,
    estimate a set of anchors using the centroids from Kmeans clustering.

    :param num_anchors: the number of anchors to return
    :param gt_wh: an array of shape [N, 2] representing the width and height of each ground truth bounding box

    """
    print(f"Running kmeans for {num_anchors} anchors on {len(gt_wh)} points...")
    std_dev = gt_wh.std(0)
    proposed_anchors, _ = kmeans(
        gt_wh / std_dev, num_anchors, iter=30
    )  # divide by std so they are in approx same range
    proposed_anchors *= std_dev

    return proposed_anchors


def anchor_fitness(anchors, wh):
    """
    A fitness function that can be used to evolve a set of anchors. This function calculates the mean best anchor ratio
    for all matches that are within the multiple range considered during the loss calculation.
    """
    best_anchor_ratio = calculate_best_anchor_ratio(anchors=anchors, gt_wh=wh)
    return (
        best_anchor_ratio
        * (best_anchor_ratio > 1 / LOSS_ANCHOR_MULTIPLE_THRESHOLD).float()
    ).mean()


def evolve_anchors(
    proposed_anchors,
    gt_wh,
    num_iterations=1000,
    mutation_probability=0.9,
    mutation_noise_mean=1,
    mutation_noise_std=0.1,
    anchor_fitness_fn=anchor_fitness,
    verbose=False,
):
    """
    Use a genetic algorithm to mutate the given anchors to try and optimise them based on the given widths and heights of the
    ground truth boxes based on the provided fitness function. Anchor dimensions are mutated by adding random noise sampled
    from a normal distribution with the mean and standard deviation provided.

    :param proposed_anchors: a tensor containing the aspect ratios of the anchor boxes to evolve
    :param gt_wh: a tensor of shape [N, 2] representing the width and height of each ground truth bounding box
    :param num_generations: the number of iterations for which to run the algorithm
    :param mutation_probability: the probability that each anchor dimension is mutated during each iteration
    :param mutation_noise_mean: the mean of the normal distribution from which the mutation noise will be sampled
    :param mutation_noise_std: the standard deviation of the normal distribution from which the mutation noise will be sampled
    :param anchor_fitness_fn: the reward function that will be used during the optimization process. This should accept proposed_anchors and gt_wh as arguments
    :param verbose: if True, the value of the fitness function will be printed at the end of each iteration

    """
    best_fitness = anchor_fitness_fn(proposed_anchors, gt_wh)
    anchor_shape = proposed_anchors.shape

    pbar = tqdm(range(num_iterations), desc=f"Evolving anchors with Genetic Algorithm:")
    for i, _ in enumerate(pbar):
        # Define mutation by sampling noise from a normal distribution
        anchor_mutation = np.ones(anchor_shape)
        anchor_mutation = (
            (np.random.random(anchor_shape) < mutation_probability)
            * np.random.randn(*anchor_shape)
            * mutation_noise_std
            + mutation_noise_mean
        ).clip(0.3, 3.0)

        mutated_anchors = (proposed_anchors.copy() * anchor_mutation).clip(min=2.0)
        mutated_anchor_fitness = anchor_fitness_fn(mutated_anchors, gt_wh)

        if mutated_anchor_fitness > best_fitness:
            best_fitness, proposed_anchors = (
                mutated_anchor_fitness,
                mutated_anchors.copy(),
            )
            pbar.desc = (
                f"Evolving anchors with Genetic Algorithm: fitness = {best_fitness:.4f}"
            )
            if verbose:
                print(f"Iteration: {i}, Fitness: {best_fitness}")

    return proposed_anchors


def sort_anchors(anchors):
    """
    Returns a sorted tensor of anchors in ascending order based on the minimum dimension for each anchor.
    """
    return torch.as_tensor(anchors)[
        torch.sort(torch.as_tensor(anchors.min(-1))).indices
    ]


def update_model_anchors(model, new_anchors):
    # TODO move inside model
    new_anchors = torch.tensor(
        new_anchors, device=model.detection_head.anchors.device
    ).type_as(model.detection_head.anchors)

    # update anchor grid used for inference
    model.detection_head.anchor_grid[:] = new_anchors.clone().view_as(
        model.detection_head.anchor_grid
    )

    # update anchors used for loss calculation
    model.detection_head.anchors[:] = new_anchors.clone().view_as(
        model.detection_head.anchors
    ) / model.detection_head.stride.to(model.detection_head.anchors.device).view(
        -1, 1, 1
    )

    check_anchor_order(model.detection_head)


def check_anchor_order(m):
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print("Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)
=======
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
        m.anchor_sizes_per_layer[:] = m.anchor_sizes_per_layer.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def update_model_anchors_inplace(model, anchors):
    # TODO move inside model
    detection_head = model.detection_head

    anchors = torch.tensor(anchors, device=detection_head.anchor_sizes_per_layer.device).type_as(
        detection_head.anchor_sizes_per_layer
    )
    detection_head.anchor_grid[:] = anchors.clone().view_as(
        detection_head.anchor_grid
    )  # for inference
    detection_head.anchor_sizes_per_layer[:] = anchors.clone().view_as(
        detection_head.anchor_sizes_per_layer
    ) / detection_head.stride.to(detection_head.anchor_sizes_per_layer.device).view(
        -1, 1, 1
    )  # loss
    check_anchor_order(detection_head)


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
>>>>>>> Model refactor seems to be working so farwq
