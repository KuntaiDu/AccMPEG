"""
    The losses here is used to train the NN-based neural network generator.
"""

from pdb import set_trace

import torch
import torch.nn.functional as F


def cross_entropy(mask, target, thresh_list):

    weight = 1.0

    # Cross entropy
    mask = mask.softmax(dim=1)[:, 1:2, :, :]
    mask = torch.where(mask < 1e-6, torch.ones_like(mask) * 1e-6, mask)
    mask = torch.where(
        mask > (1 - 1e-6), torch.ones_like(mask) * (1 - 1e-6), mask
    )
    return (
        -weight * target * mask.log() - (1 - target) * (1 - mask).log()
    ).mean()


def log_cross_entropy(mask, target, weight=1):

    target = (target + (1e-6)).log()
    target = (target - target.min()) / (target.max() - target.min())

    weight_tensor = target * (weight - 1) + target

    diff = (mask - target).abs()
    diff = torch.where(diff < 1e-6, torch.ones_like(diff) * 1e-6, diff)
    diff = torch.where(
        diff > (1 - 1e-6), torch.ones_like(diff) * (1 - 1e-6), diff
    )

    # Hope to reduce the diff, so all elements in diff should be classified as 0.
    pt = 1 - diff
    return (weight_tensor * (-((1 - pt)) * pt.log())).mean()


def cross_entropy_thresh(mask, target, thresh_list):

    loss = torch.nn.CrossEntropyLoss()
    ret = 0
    for thresh in thresh_list:
        ret = ret + loss(mask, (target > thresh).long()[:, 0, :, :])
    return ret


def mean_squared_error(mask, target, thresh_list):

    mask = mask.softmax(dim=1)[:, 1:2, :, :]
    return ((mask - target) ** 2).mean()


def focal_loss(mask, target, weight=1):

    # # focal loss makes me happy
    # target = target.float()
    # prob = mask.softmax(dim=1)[:, 1:2, :, :]
    # prob = torch.where(target == 1, prob, 1 - prob)
    # weight_tensor = torch.where(
    #     target == 1, weight * torch.ones_like(prob), torch.ones_like(prob)
    # )

    # eps = 1e-6

    # return (-weight_tensor * ((1 - prob) ** 2) * ((prob + eps).log())).mean()

    weight_tensor = target * (weight - 1) + target

    diff = (mask - target).abs()
    diff = torch.where(diff < 1e-6, torch.ones_like(diff) * 1e-6, diff)
    diff = torch.where(
        diff > (1 - 1e-6), torch.ones_like(diff) * (1 - 1e-6), diff
    )

    # Hope to reduce the diff, so all elements in diff should be classified as 0.
    pt = 1 - diff
    return (weight_tensor * (-((1 - pt) ** 2) * pt.log())).mean()
