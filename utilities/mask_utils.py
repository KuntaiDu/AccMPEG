import glob
import os
import pickle
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pdb import set_trace

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from . import bbox_utils as bu
from . import video_utils as vu
from .timer import Timer


def generate_masked_image(mask, video_slices, bws):

    masked_image = torch.zeros_like(mask)

    for i in range(len(video_slices) - 1):

        x0, x1 = bws[i], bws[i + 1]
        y0, y1 = video_slices[i], video_slices[i + 1]

        if x1 == 1:
            inds = torch.logical_and(x0 <= mask, mask <= x1)
        else:
            inds = torch.logical_and(x0 <= mask, mask < x1)

        term = y0 + (mask - x0) / (x1 - x0) * (y1 - y0)
        masked_image += torch.where(inds, term, torch.zeros_like(mask))

    return masked_image


def tile_mask(mask, tile_size):
    """
        Here the mask is of shape [1, 1, H, W]
        Eg: 
        mask = [    1   2
                    3   4]
        tile_mask(mask, 2) will return
        ret =  [    1   1   2   2
                    1   1   2   2
                    3   3   4   4
                    3   3   4   4]
        This function controlls the granularity of the mask.        
    """
    mask = mask[0, 0, :, :]
    t = tile_size
    mask = mask.unsqueeze(1).repeat(1, t, 1).view(-1, mask.shape[1])
    mask = mask.transpose(0, 1)
    mask = mask.unsqueeze(1).repeat(1, t, 1).view(-1, mask.shape[1])
    mask = mask.transpose(0, 1)
    return torch.cat(3 * [mask[None, None, :, :]], 1)


def tile_masks(mask, tile_size):
    """
        Here the mask is of shape [N, 1, H, W]
    """

    return torch.cat(
        [tile_mask(mask_slice, tile_size) for mask_slice in mask.split(1)]
    )


def mask_clip(mask, minval):
    mask.requires_grad = False
    mask[mask < minval] = minval
    mask[mask > 1] = 1
    mask.requires_grad = True


def binarize_mask(mask, bw):
    assert sorted(bw) == bw
    # assert mask.requires_grad == False

    mask_ret = mask.detach().clone()

    for i in range(len(bw) - 1):

        mid = (bw[i] + bw[i + 1]) / 2

        mask_ret[torch.logical_and(mask > bw[i], mask <= mid)] = bw[i]
        mask_ret[torch.logical_and(mask > mid, mask < bw[i + 1])] = bw[i + 1]
    return mask_ret


def generate_masked_video(mask, videos, bws, args):

    masked_video = torch.zeros_like(videos[-1])

    for fid, (video_slices, mask_slice) in enumerate(
        zip(zip(*[video.split(1) for video in videos]), mask.split(1))
    ):
        mask_slice = tile_mask(mask_slice, args.tile_size)
        masked_image = generate_masked_image(mask_slice, video_slices, bws)
        masked_video[fid : fid + 1, :, :, :] = masked_image

    return masked_video


# def encode_masked_video(args, qp, binary_mask, logger):

#     # create temp folder to write png files
#     from datetime import datetime

#     temp_folder = "temp_" + datetime.now().strftime("%m.%d.%Y,%H:%M:%S")

#     Path(temp_folder).mkdir()

#     # make a progress bar
#     progress_bar = enlighten.get_manager().counter(
#         total=binary_mask.shape[0],
#         desc=f"Generate raw png of {args.output}.qp{qp}",
#         unit="frames",
#     )

#     subprocess.run(["mkdir", args.source + ".pngs"])

#     subprocess.run(
#         [
#             "ffmpeg",
#             "-y",
#             "-f",
#             "rawvideo",
#             "-pix_fmt",
#             "yuv420p",
#             "-s:v",
#             "1280x720",
#             "-i",
#             args.source,
#             "-start_number",
#             "0",
#             args.source + ".pngs/%010d.png",
#         ]
#     )
#     # read png files from source and multiply it by mask
#     for i in range(binary_mask.shape[0]):
#         progress_bar.update()
#         source_image = Image.open((args.source + ".pngs/%010d.png") % i)
#         image_tensor = T.ToTensor()(source_image)
#         binary_mask_slice = tile_mask(binary_mask[i : i + 1, :, :, :], args.tile_size)
#         image_tensor = image_tensor * binary_mask_slice
#         dest_image = T.ToPILImage()(image_tensor[0, :, :, :])
#         dest_image.save(temp_folder + "/%010d.png" % i)

#     subprocess.run(
#         [
#             "ffmpeg",
#             "-i",
#             temp_folder + "/%010d.png",
#             "-start_number",
#             "0",
#             "-vcodec",
#             "rawvideo",
#             "-an",
#             args.output + ".yuv",
#         ]
#     )

#     # encode it through ffmpeg
#     subprocess.run(
#         [
#             "kvazaar",
#             "--input",
#             args.output + ".yuv",
#             "--input-res",
#             "1280x720",
#             "-q",
#             f"{qp}",
#             "--gop",
#             "0",
#             "--output",
#             f"{args.output}.qp{qp}",
#         ]
#     )

#     # remove temp folder
#     subprocess.run(["rm", "-r", temp_folder])

#     subprocess.run(["rm", args.output + ".yuv"])


# def write_masked_video(mask, args, qps, bws, logger):

#     mask = mask[:, 0, :, :]
#     mask = mask.permute(0, 2, 1)
#     delta_qp0 = torch.ones_like(mask) * (qps[0] - 22)
#     delta_qp1 = torch.ones_like(mask) * (qps[1] - 22)
#     mask = torch.where(mask == 0, delta_qp0, delta_qp1).int()

#     _, w, h = mask.shape

#     with open("temp.dat", "wb") as f:
#         for fid in range(len(mask)):
#             f.write(struct.pack("i", w))
#             f.write(struct.pack("i", h))
#             for j in range(h):
#                 for i in range(w):
#                     f.write(struct.pack("b", mask[fid, i, j]))

#     subprocess.run(
#         [
#             "kvazaar",
#             "--input",
#             args.source,
#             "--gop",
#             "0",
#             "--input-res",
#             "1280x720",
#             "--roi-file",
#             "temp.dat",
#             "--output",
#             args.output,
#         ]
#     )

#     with open(f"{args.output}.args", "wb") as f:
#         pickle.dump(args, f)


# def write_black_bkgd_video(mask, args, qps, bws, logger):

#     subprocess.run(["rm", "-r", args.output + "*"])

#     with open(f"{args.output}.mask", "wb") as f:
#         pickle.dump(mask, f)
#     with open(f"{args.output}.args", "wb") as f:
#         pickle.dump(args, f)

#     # slightly dilate the mask a bit, to "protect" the crucial area
#     # mask = F.conv2d(mask, torch.ones([1, 1, 3, 3]), stride=1, padding=1)
#     # mask = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))

#     os.system(f"rm -r {args.output}.source.pngs")
#     os.system(f"cp -r {args.source} {args.output}.source.pngs")

#     progress_bar = enlighten.get_manager().counter(
#         total=mask.shape[0], desc=f"Generate raw png of {args.output}", unit="frames"
#     )

#     # for mask_slice in mask.split(30):
#     #     mask_slice_mean = mask_slice.sum(dim=0, keepdim=True)
#     #     mask_slice_mean = torch.where(
#     #         mask_slice_mean > 0,
#     #         torch.ones_like(mask_slice),
#     #         torch.zeros_like(mask_slice),
#     #     )
#     #     mask_slice[:, :, :, :] = mask_slice_mean

#     with ThreadPoolExecutor(max_workers=3) as executor:
#         for fid, mask_slice in enumerate(mask.split(1)):
#             progress_bar.update()
#             # read image
#             filename = args.output + ".source.pngs/%010d.png" % fid
#             # with Timer("open", logger):
#             # with Timer("process", logger):  # 0.05s
#             # with Timer("save", logger):  # 0.1s
#             image = Image.open(filename)
#             image = T.ToTensor()(image)
#             image = image[None, :, :, :]
#             # generate background
#             mean = torch.tensor([0.485, 0.456, 0.406])
#             background = torch.ones_like(image) * mean[None, :, None, None]
#             # extract mask
#             mask_slice = tile_mask(mask_slice, args.tile_size)
#             # construct and write image
#             image = torch.where(mask_slice == 1, image, background)
#             image = T.ToPILImage()(image[0, :, :, :])
#             executor.submit(image.save, filename)

#     # assert qps[0] == 22
#     subprocess.run(
#         [
#             "ffmpeg",
#             "-y",
#             "-i",
#             args.output + ".source.pngs/%010d.png",
#             "-start_number",
#             "0",
#             "-qp",
#             f"{qps[0]}",
#             args.output,
#         ]
#     )


# def write_black_bkgd_video_smoothed(mask, args, qps, bws, logger, smooth_frames):

#     subprocess.run(["rm", "-r", args.output + "*"])

#     with open(f"{args.output}.mask", "wb") as f:
#         pickle.dump(mask, f)
#     with open(f"{args.output}.args", "wb") as f:
#         pickle.dump(args, f)

#     # slightly dilate the mask a bit, to "protect" the crucial area
#     # mask = F.conv2d(mask, torch.ones([1, 1, 3, 3]), stride=1, padding=1)
#     # mask = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))

#     os.system(f"rm -r {args.output}.source.pngs")
#     os.system(f"cp -r {args.source} {args.output}.source.pngs")

#     progress_bar = enlighten.get_manager().counter(
#         total=mask.shape[0], desc=f"Generate raw png of {args.output}", unit="frames"
#     )

#     for mask_slice in mask.split(smooth_frames):
#         mask_slice_mean = mask_slice.sum(dim=0, keepdim=True)
#         mask_slice_mean = torch.where(
#             mask_slice_mean > 0,
#             torch.ones_like(mask_slice),
#             torch.zeros_like(mask_slice),
#         )
#         mask_slice[:, :, :, :] = mask_slice_mean

#     with ThreadPoolExecutor(max_workers=3) as executor:
#         for fid, mask_slice in enumerate(mask.split(1)):
#             progress_bar.update()
#             # read image
#             filename = args.output + ".source.pngs/%010d.png" % fid
#             # with Timer("open", logger):
#             # with Timer("process", logger):  # 0.05s
#             # with Timer("save", logger):  # 0.1s
#             image = Image.open(filename)
#             image = T.ToTensor()(image)
#             image = image[None, :, :, :]
#             # generate background
#             mean = torch.tensor([0.485, 0.456, 0.406])
#             background = torch.ones_like(image) * mean[None, :, None, None]
#             # extract mask
#             mask_slice = tile_mask(mask_slice, args.tile_size)
#             # construct and write image
#             image = torch.where(mask_slice == 1, image, background)
#             image = T.ToPILImage()(image[0, :, :, :])
#             executor.submit(image.save, filename)

#     # assert qps[0] == 22
#     subprocess.run(
#         [
#             "ffmpeg",
#             "-y",
#             "-i",
#             args.output + ".source.pngs/%010d.png",
#             "-start_number",
#             "0",
#             "-qp",
#             f"{qps[0]}",
#             args.output,
#         ]
#     )


def dilate_binarize(mask, lower_bound, kernel_size, cuda=True):
    kernel = torch.ones([1, 1, kernel_size, kernel_size])
    if cuda:
        kernel = kernel.cuda(non_blocking=True)
    mask = torch.where(
        (mask > lower_bound), torch.ones_like(mask), torch.zeros_like(mask),
    )
    mask = F.conv2d(mask, kernel, stride=1, padding=(kernel_size - 1) // 2,)
    mask = torch.where(
        mask > 0.5, torch.ones_like(mask), torch.zeros_like(mask),
    )
    return mask


def write_black_bkgd_video_smoothed_continuous(
    mask, args, qp, logger, protect=False, writer=None, tag=None
):

    subprocess.run(["rm", "-r", args.output + "*"])

    with open(f"{args.output}.args", "wb") as f:
        pickle.dump(args, f)

    # slightly dilate the mask a bit, to "protect" the crucial area
    # mask = F.conv2d(mask, torch.ones([1, 1, 3, 3]), stride=1, padding=1)
    # mask = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))

    logger.info("Copying source images...")

    os.system(f"rm -r {args.output}.source.pngs")
    os.system(f"cp -r {args.source} {args.output}.source.pngs")

    # for mask_slice in mask.split(args.smooth_frames):
    #     mask_slice[:, :, :, :] = mask_slice.mean(dim=0, keepdim=True)

    # if hasattr(args, "upper_bound") and hasattr(args, "lower_bound"):
    #     logger.info("Using upper bound and lower bound.")
    #     # mask = torch.where(
    #     #     (mask < args.upper_bound) & (mask >= args.lower_bound),
    #     #     torch.ones_like(mask),
    #     #     torch.zeros_like(mask),
    #     # )
    #     # mask = dilate_binarize(mask.cuda(), 0.5, args.conv_size).cpu()
    #     assert args.upper_bound >= args.lower_bound
    #     mask = mask.cuda()
    #     x = dilate_binarize(mask, args.lower_bound, args.conv_size)
    #     y = dilate_binarize(mask, args.upper_bound, args.conv_size)
    #     # set_trace()
    #     mask = x - y
    #     mask = mask.cpu()
    # else:
    #     logger.info("Using single bound.")
    #     if hasattr(args, "conv_size_large") and args.conv_size_large != -1:
    #         mask = mask.cuda()
    #         maska = dilate_binarize(mask, args.bound, args.conv_size).cpu()
    #         maskb = dilate_binarize(mask, args.bound, args.conv_size_large).cpu()
    #         mask = maskb - maska
    #     else:
    #         mask = dilate_binarize(mask.cuda(), args.bound, args.conv_size).cpu()

    # set_trace()

    assert ((mask == 0) | (mask == 1)).all()

    with open(f"{args.output}.mask", "wb") as f:
        pickle.dump(mask, f)

    if protect:
        mask = dilate_binarize(mask, 0.5, 3, False)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for fid, mask_slice in enumerate(tqdm(mask.split(1))):
            # read image
            filename = args.output + ".source.pngs/%010d.png" % fid
            # with Timer("open", logger):
            # with Timer("process", logger):  # 0.05s
            # with Timer("save", logger):  # 0.1s
            image = Image.open(filename)
            image = T.ToTensor()(image)
            image = image[None, :, :, :]
            # generate background
            mean = torch.Tensor([0.485, 0.456, 0.406])
            # mean = torch.Tensor([0.0, 0.0, 0.0])
            background = torch.ones_like(image) * mean[None, :, None, None]
            # extract mask
            mask_slice = tile_mask(mask_slice, args.tile_size)
            # construct and write image
            image = torch.where(mask_slice == 1, image, background)
            if writer is not None and fid % args.visualize_step_size == 0:
                assert tag is not None, "Please assign a tag for the writer"
                writer.add_image(tag, image[0], fid)
            image = T.ToPILImage()(image[0])
            executor.submit(image.save, filename)

    # assert qps[0] == 22
    file_extension = args.output.split(".")[-1]

    if file_extension == "mp4":
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                args.output + ".source.pngs/%010d.png",
                "-start_number",
                "0",
                "-qmin",
                f"{qp}",
                "-qmax",
                f"{qp}",
                args.output,
            ]
        )
    elif file_extension == "hevc":
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                args.output + ".source.pngs/%010d.png",
                "-start_number",
                "0",
                "-c:v",
                "libx265",
                "-x265-params",
                f"qp={qp}",
                args.output,
            ]
        )
    elif file_extension == "webm":
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                args.output + ".source.pngs/%010d.png",
                "-start_number",
                "0",
                "-c:v",
                "libvpx-vp9",
                "-crf",
                f"{qp}",
                "-b:v",
                "0",
                "-threads",
                "8",
                args.output,
            ]
        )

    os.system(f"rm -r {args.output}.source.pngs")


def write_black_bkgd_video_smoothed_continuous_crf(
    mask, args, qp, logger, protect=False, writer=None, tag=None
):

    subprocess.run(["rm", "-r", args.output + "*"])

    with open(f"{args.output}.args", "wb") as f:
        pickle.dump(args, f)

    # slightly dilate the mask a bit, to "protect" the crucial area
    # mask = F.conv2d(mask, torch.ones([1, 1, 3, 3]), stride=1, padding=1)
    # mask = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))

    os.system(f"rm -r {args.output}.source.pngs")
    os.system(f"cp -r {args.source} {args.output}.source.pngs")

    # for mask_slice in mask.split(args.smooth_frames):
    #     mask_slice[:, :, :, :] = mask_slice.mean(dim=0, keepdim=True)

    # if hasattr(args, "upper_bound") and hasattr(args, "lower_bound"):
    #     logger.info("Using upper bound and lower bound.")
    #     # mask = torch.where(
    #     #     (mask < args.upper_bound) & (mask >= args.lower_bound),
    #     #     torch.ones_like(mask),
    #     #     torch.zeros_like(mask),
    #     # )
    #     # mask = dilate_binarize(mask.cuda(), 0.5, args.conv_size).cpu()
    #     assert args.upper_bound >= args.lower_bound
    #     mask = mask.cuda()
    #     x = dilate_binarize(mask, args.lower_bound, args.conv_size)
    #     y = dilate_binarize(mask, args.upper_bound, args.conv_size)
    #     # set_trace()
    #     mask = x - y
    #     mask = mask.cpu()
    # else:
    #     logger.info("Using single bound.")
    #     if hasattr(args, "conv_size_large") and args.conv_size_large != -1:
    #         mask = mask.cuda()
    #         maska = dilate_binarize(mask, args.bound, args.conv_size).cpu()
    #         maskb = dilate_binarize(mask, args.bound, args.conv_size_large).cpu()
    #         mask = maskb - maska
    #     else:
    #         mask = dilate_binarize(mask.cuda(), args.bound, args.conv_size).cpu()

    # set_trace()

    assert ((mask == 0) | (mask == 1)).all()

    with open(f"{args.output}.mask", "wb") as f:
        pickle.dump(mask, f)

    if protect:
        mask = dilate_binarize(mask, 0.5, 3, False)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for fid, mask_slice in enumerate(tqdm(mask.split(1))):
            # read image
            filename = args.output + ".source.pngs/%010d.png" % fid
            # with Timer("open", logger):
            # with Timer("process", logger):  # 0.05s
            # with Timer("save", logger):  # 0.1s
            image = Image.open(filename)
            image = T.ToTensor()(image)
            image = image[None, :, :, :]
            # generate background
            # mean = torch.Tensor([0.485, 0.456, 0.406])
            mean = torch.Tensor([0.0, 0.0, 0.0])
            background = torch.ones_like(image) * mean[None, :, None, None]
            # extract mask
            mask_slice = tile_mask(mask_slice, args.tile_size)
            # construct and write image
            image = torch.where(mask_slice == 1, image, background)
            if writer is not None and fid % args.visualize_step_size == 0:
                assert tag is not None, "Please assign a tag for the writer"
                writer.add_image(tag, image[0], fid)
            image = T.ToPILImage()(image[0])
            executor.submit(image.save, filename)

    # assert qps[0] == 22
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            args.output + ".source.pngs/%010d.png",
            "-start_number",
            "0",
            "-c:v",
            "libx264",
            "-x264-params",
            "nal-hrd=cbr",
            "-b:v",
            f"{qp}M",
            "-minrate",
            f"{qp}M",
            "-maxrate",
            f"{qp}M",
            "-bufsize",
            "2M",
            args.output,
        ]
    )


# def write_black_bkgd_video_smoothed(mask, args, qps, bws, logger):

#     subprocess.run(["rm", "-r", args.output + "*"])

#     with open(f"{args.output}.mask", "wb") as f:
#         pickle.dump(mask, f)
#     with open(f"{args.output}.args", "wb") as f:
#         pickle.dump(args, f)

#     # slightly dilate the mask a bit, to "protect" the crucial area
#     # mask = F.conv2d(mask, torch.ones([1, 1, 3, 3]), stride=1, padding=1)
#     # mask = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))

#     Path(args.output + ".source.pngs").mkdir(exist_ok=True)
#     subprocess.run(
#         [
#             "ffmpeg",
#             "-y",
#             "-f",
#             "rawvideo",
#             "-pix_fmt",
#             "yuv420p",
#             "-s:v",
#             "1280x720",
#             "-i",
#             args.source,
#             "-start_number",
#             "0",
#             args.output + ".source.pngs/%010d.png",
#         ]
#     )

#     progress_bar = enlighten.get_manager().counter(
#         total=mask.shape[0], desc=f"Generate raw png of {args.output}", unit="frames"
#     )

#     for mask_slice in mask.split(30):
#         mask_slice_mean = mask_slice.sum(dim=0, keepdim=True)
#         mask_slice_mean = torch.where(
#             mask_slice_mean > 0,
#             torch.ones_like(mask_slice),
#             torch.zeros_like(mask_slice),
#         )
#         mask_slice[:, :, :, :] = mask_slice_mean

#     for fid, mask_slice in enumerate(mask.split(1)):
#         progress_bar.update()
#         # read image
#         filename = args.output + ".source.pngs/%010d.png" % fid
#         image = Image.open(filename)
#         image = T.ToTensor()(image)
#         image = image[None, :, :, :]
#         # generate background
#         mean = torch.tensor([0.485, 0.456, 0.406])
#         background = torch.ones_like(image) * mean[None, :, None, None]
#         # extract mask
#         mask_slice = tile_mask(mask_slice, args.tile_size)
#         # construct and write image
#         image = torch.where(mask_slice == 1, image, background)
#         T.ToPILImage()(image[0, :, :, :]).save(filename)

#     subprocess.run(
#         [
#             "ffmpeg",
#             "-y",
#             "-i",
#             args.output + ".source.pngs/%010d.png",
#             "-start_number",
#             "0",
#             "-vcodec",
#             "rawvideo",
#             "-pix_fmt",
#             "yuv420p",
#             args.output + ".yuv",
#         ]
#     )

#     assert qps[0] == 22

#     subprocess.run(
#         [
#             "kvazaar",
#             "--input",
#             args.output + ".yuv",
#             "--input-res",
#             "1280x720",
#             "-q",
#             f"{qps[0]}",
#             "--gop",
#             "0",
#             "--output",
#             args.output,
#         ]
#     )


# def read_masked_video(video_name, logger):

#     logger.info(f"Reading compressed video {video_name}. Reading each part...")
#     parts = sorted(glob.glob(f"{video_name}.qp[0-9]*"), reverse=True)
#     parts2 = []

#     videos = []
#     # import pdb; pdb.set_trace()
#     # parts2 = ['youtube_videos/train_first/dashcam_1_train_qp_%d.mp4' % i for i in [24, 38]]
#     for part in parts:
#         videos.append(vu.read_video(part, logger))
#     logger.info(f"Reading mask for compressed video.")

#     with open(f"{video_name}.mask", "rb") as f:
#         filename2mask = pickle.load(f)
#     tile_size = filename2mask["args.tile_size"]

#     base = videos[0]
#     for video, part in zip(videos[1:], parts[1:]):
#         video = video * tile_masks(filename2mask[part], tile_size)
#         base[video != 0] = video[video != 0]
#     return base


def generate_mask_from_regions(
    mask_slice, regions, minval, tile_size, cuda=False
):

    # (xmin, ymin, xmax, ymax)
    regions = bu.point_form(regions)
    mask_slice[:, :, :, :] = minval
    mask_slice_orig = mask_slice

    # tile the mask
    mask_slice = tile_mask(mask_slice, tile_size)

    # put regions on it
    x = mask_slice.shape[3]
    y = mask_slice.shape[2]

    for region in regions:
        xrange = torch.arange(0, x)
        yrange = torch.arange(0, y)

        xmin, ymin, xmax, ymax = region
        yrange = (yrange >= ymin) & (yrange <= ymax)
        xrange = (xrange >= xmin) & (xrange <= xmax)

        if xrange.nonzero().nelement() == 0 or yrange.nonzero().nelement() == 0:
            continue

        xrangemin = xrange.nonzero().min().item()
        xrangemax = xrange.nonzero().max().item() + 1
        yrangemin = yrange.nonzero().min().item()
        yrangemax = yrange.nonzero().max().item() + 1
        mask_slice[:, :, yrangemin:yrangemax, xrangemin:xrangemax] = 1

    # revert the tile process
    mask_slice = F.conv2d(
        mask_slice,
        torch.ones([1, 3, tile_size, tile_size]).cuda()
        if cuda
        else torch.ones([1, 3, tile_size, tile_size]),
        stride=tile_size,
    )
    mask_slice = torch.where(
        mask_slice > 0.5,
        torch.ones_like(mask_slice),
        torch.zeros_like(mask_slice),
    )
    mask_slice_orig[:, :, :, :] = mask_slice[:, :, :, :]

    return mask_slice_orig


def percentile(t: torch.tensor, q: float) -> float:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(0.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def merge_black_bkgd_images(images, mask, args):

    images = [F.interpolate(image, size=(720, 1280)) for image in images]
    mask = tile_mask(mask, args.tile_size)

    return torch.where(mask == 1, images[1], images[0])


def postprocess_mask(mask, kernel_size=3):

    assert ((mask == 0) | (mask == 1)).all()
    eps = 1e-5
    kernel = torch.ones([1, 1, kernel_size, kernel_size])

    # remove small noises
    mask = F.conv2d(mask, kernel, stride=1, padding=(kernel_size - 1) // 2,)
    mask = ((mask - (kernel_size * kernel_size)).abs() < eps).float()
    mask = F.conv2d(mask, kernel, stride=1, padding=(kernel_size - 1) // 2,)
    mask = (mask > eps).float()

    # fill small holes
    mask = F.conv2d(mask, kernel, stride=1, padding=(kernel_size - 1) // 2,)
    mask = (mask > eps).float()
    mask = F.conv2d(mask, kernel, stride=1, padding=(kernel_size - 1) // 2,)
    mask = ((mask - (kernel_size * kernel_size)).abs() < eps).float()

    return mask
