import io
from pathlib import Path
from pdb import set_trace

import matplotlib.pyplot as plt
import PIL
import seaborn as sns
import torchvision.transforms as T

from .mask_utils import tile_mask


def visualize_heat(image, heat, path, args, overwrite=True, tile=True):
    if Path(path).exists() and not overwrite:
        return

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
    if tile:
        heat = tile_mask(heat, args.tile_size,)[0, 0, :, :]
    else:
        heat = heat[0, 0, :, :]
    heat[heat != heat] = 0
    ax = sns.heatmap(
        heat.cpu().detach().numpy(),
        zorder=3,
        alpha=0.5,
        ax=ax,
        xticklabels=False,
        yticklabels=False,
    )
    ax.imshow(image, zorder=3, alpha=0.5)
    ax.tick_params(left=False, bottom=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def visualize_dist(heat, path, overwrite=True):
    if Path(path).exists() and not overwrite:
        return

    heat[heat != heat] = 0

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
    ax = sns.distplot(heat.flatten().detach().numpy(), ax=ax)
    # ax.tick_params(left=False, bottom=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def visualize_scores2grads(scores2grads, path, overwrite=True):
    if Path(path).exists() and not overwrite:
        return

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
    ax.scatter([i[0] for i in scores2grads], [i[1] for i in scores2grads])
    # ax.tick_params(left=False, bottom=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def visualize_dist_by_summarywriter(heat, tag, writer, fid):

    heat[heat != heat] = 0

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
    ax = sns.distplot(heat.flatten().detach().numpy(), ax=ax)
    # ax.tick_params(left=False, bottom=False)
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    result = PIL.Image.open(buf)
    writer.add_image(tag, T.ToTensor()(result), fid)


def visualize_log_dist(heat, path, overwrite=True):
    if Path(path).exists() and not overwrite:
        return

    heat[heat != heat] = 0
    heat = heat + 1e-40
    heat = heat.log()

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
    ax = sns.distplot(heat.flatten().detach().numpy(), ax=ax)
    # ax.tick_params(left=False, bottom=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def visualize_heat_by_summarywriter(
    image, heat, tag, writer, fid, args, tile=True, alpha=0.5
):

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
    if tile:
        heat = tile_mask(heat, args.tile_size,)[0, 0, :, :]
    else:
        heat = heat[0, 0, :, :]
    ax = sns.heatmap(
        heat.cpu().detach().numpy(),
        zorder=3,
        alpha=alpha,
        ax=ax,
        xticklabels=False,
        yticklabels=False,
    )
    ax.imshow(image, zorder=3, alpha=(1 - alpha))
    ax.tick_params(left=False, bottom=False)
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    result = PIL.Image.open(buf)
    writer.add_image(tag, T.ToTensor()(result), fid)
