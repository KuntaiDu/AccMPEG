
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pdb import set_trace
from tqdm import tqdm
from pathlib import Path
from shutil import rmtree, copytree
from utils.compressor import *
import pickle
import torch
import torchvision.transforms as T
from PIL import Image
from utils.mask_utils import tile_mask

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

    # generate pngs
    logger.info(f"Generate source pngs for {args.output}")
    with ThreadPoolExecutor(max_workers=4) as executor:
        for fid, mask_slice in enumerate(tqdm(mask.split(1))):
            # set filename
            input_filename = args.source + "/%010d.png" % fid
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
            "-loglevel", "warning",
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