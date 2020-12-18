"""
    The losses here is used to train the NN-based neural network generator.
"""

import torch
import torch.nn.functional as F


def cross_entropy(mask, target, weight=1.0):

    # Cross entropy
    target = target[:, 0, :, :]
    return F.cross_entropy(mask, target, torch.tensor([1.0, weight]).cuda())


def mean_squared_error(mask, target):

    target = target.float()
    prob = mask.softmax(dim=1)[:, 1:2, :, :]
    return ((target - prob) ** 2).mean()


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
