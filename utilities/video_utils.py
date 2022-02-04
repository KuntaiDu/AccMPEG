import glob
import os
import subprocess
from pathlib import Path
from pdb import set_trace

import av
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision import io

from . import mask_utils as mu


class Video(Dataset):
    def __init__(self, video, postprocess, logger, return_fid=False):
        self.video = video
        logger.info(f"Extract {video} to pngs.")
        Path(f"{video}.pngs").mkdir(exist_ok=True)
        subprocess.run(["rm", f"{video}.pngs/*.png"], stderr=subprocess.DEVNULL)
        subprocess.check_output(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-stats",
                "-y",
                "-i",
                f"{video}",
                "-start_number",
                "0",
                f"{video}.pngs/%010d.png",
            ]
        )
        self.nimages = len(glob.glob(f"{video}.pngs/*.png"))
        self.postprocess = postprocess
        self.return_fid = return_fid

    def __len__(self):
        return self.nimages

    def __getitem__(self, idx):
        image = T.ToTensor()(plt.imread(f"{self.video}.pngs/%010d.png" % idx))
        image_post = self.postprocess(image, idx)
        # # just for visualization purpose
        # if image is not image_post:
        #     T.ToPILImage()(image_post).save(f'{self.video}.pngs/%010d.png' % idx)
        if self.return_fid:
            return {"image": image_post, "fid": idx, "video_name": self.video}
        else:
            return image_post


def read_videos(
    video_list,
    logger,
    sort=False,
    normalize=True,
    dataloader=True,
    from_source=False,
):
    """
        Read a list of video and return two lists. 
        One is the video tensors, the other is the bandwidths.
    """
    video_list = [
        {
            "video": read_video(video_name, logger, dataloader, from_source),
            "bandwidth": read_bandwidth(video_name),
            "name": video_name,
        }
        for video_name in video_list
    ]
    if sort:
        video_list = sorted(video_list, key=lambda x: x["bandwidth"])

    # bandwidth normalization
    gt_bandwidth = max(video["bandwidth"] for video in video_list)
    if normalize:
        for i in video_list:
            i["bandwidth"] /= gt_bandwidth

    return (
        [i["video"] for i in video_list],
        [i["bandwidth"] for i in video_list],
        [i["name"] for i in video_list],
    )


def read_video(video_name, logger, dataloader, from_source):
    logger.info(f"Reading {video_name}")
    postprocess = lambda x, fid: x
    if "black" in video_name and "base" not in video_name:
        import pickle

        with open(f"{video_name}.mask", "rb") as f:
            mask = pickle.load(f)
        with open(f"{video_name}.args", "rb") as f:
            args = pickle.load(f)
        if from_source:
            # directly copy-paste the high quality video for high quality regions.
            if hasattr(args, "input"):
                video_name = args.input[0]
            elif hasattr(args, "inputs"):
                video_name = args.inputs[-1]
            else:
                raise RuntimeError(
                    "Cannot reason the high-quality video name from the args."
                )
        postprocess = lambda x, fid: postprocess_black_bkgd(fid, x, mask, args)
    # import pdb; pdb.set_trace()
    if dataloader:
        return DataLoader(
            Video(video_name, postprocess, logger), shuffle=False, num_workers=2
        )
    else:
        # need to return fid
        return Video(video_name, postprocess, logger, return_fid=True)


def read_videos_pyav(
    video_list,
    logger,
    sort=False,
    normalize=True,
    dataloader=True,
    from_source=False,
):
    """
        Read a list of video and return two lists. 
        One is the video tensors, the other is the bandwidths.
    """
    video_list = [
        {
            "video": read_video_pyav(
                video_name, logger, dataloader, from_source
            ),
            "bandwidth": read_bandwidth(video_name),
            "name": video_name,
        }
        for video_name in video_list
    ]
    if sort:
        video_list = sorted(video_list, key=lambda x: x["bandwidth"])

    # bandwidth normalization
    gt_bandwidth = max(video["bandwidth"] for video in video_list)
    if normalize:
        for i in video_list:
            i["bandwidth"] /= gt_bandwidth

    return (
        [i["video"] for i in video_list],
        [i["bandwidth"] for i in video_list],
        [i["name"] for i in video_list],
    )


def read_video_pyav(video_name, logger, dataloader=None, from_source=None):
    logger.info(f"Reading {video_name} (with PyAv)")
    postprocess = lambda x, fid: x
    if "black" in video_name and "base" not in video_name:
        import pickle

        with open(f"{video_name}.mask", "rb") as f:
            mask = pickle.load(f)
        with open(f"{video_name}.args", "rb") as f:
            args = pickle.load(f)
        if from_source:
            # directly copy-paste the high quality video for high quality regions.
            if hasattr(args, "input"):
                video_name = args.input[0]
            elif hasattr(args, "inputs"):
                video_name = args.inputs[-1]
            else:
                raise RuntimeError(
                    "Cannot reason the high-quality video name from the args."
                )
        postprocess = lambda x, fid: postprocess_black_bkgd(fid, x, mask, args)
    return av.open(video_name)


def postprocess_black_bkgd(fid, image, mask, args):

    mean = torch.tensor([0.485, 0.456, 0.406])
    image = image[None, :, :, :]
    background = torch.ones_like(image) * mean[None, :, None, None]
    mask_fid = mask[fid : fid + 1, :, :, :]
    mask_fid = mu.tile_mask(mask_fid, args.tile_size)
    return torch.where(mask_fid == 1, image, background)[0, :, :, :]


def read_bandwidth(video_name):
    if "dual" not in video_name:
        return os.path.getsize(video_name)
    else:
        ext = video_name.split(".")[-1]

        return sum(
            os.path.getsize(i) for i in glob.glob(video_name + f"*.{ext}")
        )


def write_video(video_tensor, video_name, logger):

    logger.info(f"Saving {video_name}")

    # [N, C, H, W] ==> [N, H, W, C]
    video_tensor = video_tensor.permute(0, 2, 3, 1)
    # go back to original domain
    video_tensor = (
        video_tensor.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
    )
    # lossless encode. Should be replaced
    io.write_video(video_name, video_tensor, fps=25, options={"crf": "0"})


def get_qp_from_name(video_name):

    # the video name format must be xxxxxxx_{qp}.mp4
    return int(video_name.split(".")[-2].split("_")[-1])

