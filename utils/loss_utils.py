"""
    The losses here is used to train the NN-based neural network generator.
"""

import torch
import torch.nn.functional as F


def cross_entropy(mask, target, weight=1.0):

    # Cross entropy
    mask = torch.where(mask < 1e-6, torch.ones_like(mask) * 1e-6, mask)
    mask = torch.where(mask > (1 - 1e-6), torch.ones_like(mask) * (1 - 1e-6), mask)
    return (-weight * target * mask.log() - (1 - target) * (1 - mask).log()).mean()


def mean_squared_error(mask, target):

    return (mask - target).norm(p=2)


def focal_loss(mask, target, weight=1):

    # focal loss makes me happy
    target = target.float()
    prob = mask.softmax(dim=1)[:, 1:2, :, :]
    prob = torch.where(target == 1, prob, 1 - prob)
    weight_tensor = torch.where(
        target == 1, weight * torch.ones_like(prob), torch.ones_like(prob)
    )

    eps = 1e-6

    return (-weight_tensor * ((1 - prob) ** 2) * ((prob + eps).log())).mean()
