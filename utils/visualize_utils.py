from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from .mask_utils import tile_mask


def visualize_heat(image, heat, path, args, overwrite=True):
    if Path(path).exists() and not overwrite:
        return

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
    heat = tile_mask(heat, args.tile_size,)[0, 0, :, :]
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

