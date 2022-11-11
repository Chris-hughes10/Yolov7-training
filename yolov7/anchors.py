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
