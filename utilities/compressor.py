import glob
import os
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pdb import set_trace
from shutil import copytree, rmtree
from time import sleep

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from config import settings

# from utils.compressor import *
from utilities.mask_utils import tile_mask


def black_background_compressor(mask, args, logger, writer):

    # cleanup previous results
    subprocess.run(["rm", "-r", args.output + "*"])
    if Path(f"{args.output}.source.pngs").exists():
        rmtree(f"{args.output}.source.pngs")
    Path(f"{args.output}.source.pngs").mkdir()

    # dump args for decoding purpose.
    with open(f"{args.output}.args", "wb") as f:
        pickle.dump(args, f)

    # mask must be binary
    assert ((mask == 0) | (mask == 1)).all()

    with open(f"{args.output}.mask", "wb") as f:
        pickle.dump(mask, f)

    # uniform color background
    mean = torch.Tensor([0.0, 0.0, 0.0])
    background = None

    # generate source pngs
    subprocess.run(["rm", "-r", f"{args.source}.pngs"])
    Path(f"{args.source}.pngs").mkdir()
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-stats",
            "-y",
            "-i",
            args.source,
            "-start_number",
            "0",
            f"{args.source}.pngs/%010d.png",
        ]
    )

    # construct black-background pngs based on mask and source pngs
    logger.info(f"Generate source pngs for {args.output}")
    with ThreadPoolExecutor(max_workers=4) as executor:
        for fid, mask_slice in enumerate(tqdm(mask.split(1))):
            # set filename
            input_filename = args.source + ".pngs/%010d.png" % fid
            output_filename = args.output + ".source.pngs/%010d.png" % fid

            # read image
            image = T.ToTensor()(Image.open(input_filename)).unsqueeze(0)

            # extract mask
            mask_slice = tile_mask(mask_slice, args.tile_size)

            # construct uniform color background
            if background is None:
                background = torch.ones_like(image) * mean[None, :, None, None]

            # construct and write image
            image = torch.where(mask_slice == 1, image, background)
            if writer is not None and fid % args.visualize_step_size == 0:
                writer.add_image("before_encode", image[0], fid)
            image = T.ToPILImage()(image[0])
            executor.submit(image.save, output_filename)

    # pngs ==> mp4
    logger.info(f"Gernerate compressed video {args.output}")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-stats",
            "-y",
            "-i",
            args.output + ".source.pngs/%010d.png",
            "-start_number",
            "0",
            "-qp",
            f"{args.qp}",
            args.output,
        ]
    )

    if not args.preserve:
        rmtree(f"{args.output}.source.pngs")


def h264_compressor_segment(args, logger):

    num_pngs = len(glob.glob(args.source + "/*.png"))

    filenames = ""

    # write individual videos
    logger.info(
        "Compressing individual videos from %s in QP %d", args.source, args.qp
    )

    for idx, slice in enumerate(
        torch.split(torch.tensor(range(num_pngs)), args.smooth_frames)
    ):
        st = slice[0]
        ed = slice[-1]
        length = ed - st + 1
        print(f"{st} {ed} {length}")

        filename = f"{args.source}_qp_{args.qp}_part_{idx}.mp4"

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-stats",
                "-y",
                "-start_number",
                f"{st}",
                "-i",
                f"{args.source}/%010d.png",
                "-frames:v",
                f"{length}",
                "-qmin",
                f"{args.qp}",
                "-qmax",
                f"{args.qp}",
                filename,
            ]
        )

        filename = Path(filename).resolve()

        filenames += f"file '{filename}'\n"

    logger.info("Merging video to %s", f"{args.source}_qp_{args.qp}.mp4")

    # concat the video clips
    with open(f"{args.source}_qp_{args.qp}.txt", "w") as f:
        f.write(filenames)

    # ffmpeg -f concat -safe 0 -i vidlist.txt -c copy output

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-stats",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            f"{args.source}_qp_{args.qp}.txt",
            "-c",
            "copy",
            f"{args.source}_qp_{args.qp}.mp4",
        ]
    )

    os.system(f"rm -r {args.source}_qp_{args.qp}_part_*.mp4")
    os.system(f"rm {args.source}_qp_{args.qp}.txt")


def h264_compressor_cloudseg_segment(args, logger):

    num_pngs = len(glob.glob(args.source + "/*.png"))

    filenames = ""

    # write individual videos
    logger.info(
        "Compressing individual videos from %s in QP %d", args.source, args.qp
    )

    for idx, slice in enumerate(
        torch.split(torch.tensor(range(num_pngs)), args.smooth_frames)
    ):
        st = slice[0]
        ed = slice[-1]
        length = ed - st + 1
        print(f"{st} {ed} {length}")

        # filename = f"{args.source}_cloudseg_qp_{args.qp}_part_{idx}.mp4"
        filename = f"{args.output}.part_{idx}.mp4"

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-stats",
                "-y",
                "-start_number",
                f"{st}",
                "-i",
                f"{args.source}/%010d.png",
                "-vf",
                "scale=640:360",
                "-frames:v",
                f"{length}",
                "-qmin",
                f"{args.qp}",
                "-qmax",
                f"{args.qp}",
                filename,
            ]
        )

        filename = Path(filename).resolve()

        filenames += f"file '{filename}'\n"

    logger.info("Merging video to %s", args.output)

    # concat the video clips
    with open(f"{args.output}.txt", "w") as f:
        f.write(filenames)

    # ffmpeg -f concat -safe 0 -i vidlist.txt -c copy output

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-stats",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            f"{args.output}.txt",
            "-c",
            "copy",
            args.output,
        ]
    )

    os.system(f"rm {args.output}.part_*.mp4")
    os.system(f"rm {args.output}.txt")


def h264_roi_compressor(mask, args, logger):

    mask = mask.squeeze(1)

    logger.info("Dumping roi files...")

    with open("/tank/kuntai/code/qp_matrix_file", "w") as qp_file:

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                for k in range(mask.shape[2]):
                    qp_file.write(f"{mask[i,j,k]} ")
                qp_file.write("\n")

    logger.info("Encoding...")

    ffmpeg_env = os.environ.copy()
    ffmpeg_env["LD_LIBRARY_PATH"] = "/tank/kuntai/lib/"

    subprocess.run(
        [
            "/tank/kuntai/myh264/ffmpeg-3.4.8/ffmpeg",
            "-y",
            "-i",
            args.source + "/%010d.png",
            "-start_number",
            "0",
            args.output,
        ],
        env=ffmpeg_env,
    )


def h264_roi_compressor_segment(mask_full, args, logger):

    x264_dir = settings.x264_dir

    mask_full = mask_full.squeeze(1)
    num_pngs = mask_full.shape[0]
    ffmpeg_env = os.environ.copy()
    ffmpeg_env["LD_LIBRARY_PATH"] = f"{x264_dir}/lib"

    filenames = ""

    while os.path.exists("encoding.lock"):
        print("waiting for encoding finish")
        sleep(10)

    os.system("touch encoding.lock")

    for idx, slice in enumerate(
        torch.split(torch.tensor(range(num_pngs)), args.smooth_frames)
    ):

        st = slice[0]
        ed = slice[-1]
        length = ed - st + 1

        mask = mask_full[st : ed + 1, :, :]
        logger.info("Encoding segment %d...", idx)

        with open(f"{x264_dir}/qp_matrix_file", "w") as qp_file:

            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    for k in range(mask.shape[2]):
                        qp_file.write(f"{mask[i,j,k]} ")
                    qp_file.write("\n")

        filename = args.output + f".part_{idx}.mp4"

        subprocess.run(
            [
                f"{x264_dir}/ffmpeg-3.4.8/ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-stats",
                "-y",
                "-start_number",
                f"{st}",
                "-i",
                args.source + "/%010d.png",
                "-frames:v",
                f"{length}",
                filename,
            ],
            env=ffmpeg_env,
        )

        filename = str(Path(filename).resolve())
        filenames += f"file '{filename}'\n"

    with open(f"{args.output}.txt", "w") as f:
        f.write(filenames)

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-stats",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            f"{args.output}.txt",
            "-c",
            "copy",
            args.output,
        ]
    )

    # cleanup
    os.system(f"rm {args.output}.txt")
    os.system(f"rm {args.output}.part_*.mp4")
    os.system("rm encoding.lock")
