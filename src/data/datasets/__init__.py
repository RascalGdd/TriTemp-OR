# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .OR import build as build_or


def build_dataset(image_set, args):
    return build_or(image_set, args)