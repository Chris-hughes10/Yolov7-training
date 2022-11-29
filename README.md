# Yolov7-training

A clean, modular implementation of the Yolov7 model family, which uses the [official pretrained weights](https://github.com/WongKinYiu/yolov7), with utilities for training the model on custom (non-COCO) tasks.

This Repo includes:
- PyTorch implementations of the Yolov7 models defined in the original paper
- An implementation of the Yolov7 loss function
- Components to create a modular data loading pipeline, such as a PyTorch dataset wrapper and custom Mosaic and Mixup implementations
- Evaluation utilities using PyCOCOTools
- A Generic model Trainer, which can be extended or adapted to new tasks

Examples of use are available [here](./examples/), and more detail is provided in [this blog post](https://towardsdatascience.com/yolov7-a-deep-dive-into-the-current-state-of-the-art-for-object-detection-ce3ffedeeaeb?source=friends_link&sk=4281bc61b8197368d1092d8b8d6ffa64).

## Installation

This repo can be installed as a package using the following command:

```
pip install git+https://github.com/Chris-hughes10/Yolov7-training.git
```

## Usage

Whilst we would recommend using the official implementation if you wish to exactly reproduce the published results on COCO, we find that this implementation is more flexible to apply, and extend, to custom domains. 

The aim of this implementation is provide a clean and clear starting point for anyone wishing to experiment with Yolov7 in their own custom training scripts, as well as providing more transparency around the techniques that were used during training in the original implementation.
