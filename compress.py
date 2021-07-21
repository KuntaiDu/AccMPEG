
import pickle
import argparse

from matplotlib.pyplot import streamplot
import torch
from utils.mask_utils import percentile, dilate_binarize
import logging
import coloredlogs
import utils.compressor as compressor
from pdb import set_trace
from torch.utils.tensorboard import SummaryWriter



def main(args):

    # define logger
    logger = logging.getLogger("compress")

    
    with open(f"{args.input}.mask", "rb") as f:
        mask = pickle.load(f)
        
    splitted_mask = mask.split(args.smooth_frames)

    # for mask_slice in mask.split(args.smooth_frames):
    #     mask_slice[:, :, :, :] = mask_slice.mean(dim=0, keepdim=True)
    
    # smooth the mask first. 
    # we only sample the mask from the first image of every video segment.
    for i in range(len(splitted_mask)):
        cur_slice = splitted_mask[i]
        if i < len(splitted_mask) - 1:
            next_slice = splitted_mask[i+1]
            cur_slice[:, :, :, :] = 0.5 * (cur_slice[0]  + next_slice[0]).unsqueeze(0)
        else:
            cur_slice[:, :, :, :] = 0.5 * (cur_slice[0] + cur_slice[-1]).unsqueeze(0)
            
    # # Two types of knobs: heat value threshold and knobs
    if args.bound:
        mask = dilate_binarize(mask, args.bound, args.pad, cuda=False)
    else:
        mask = dilate_binarize(mask, percentile(mask, args.perc), args.pad, cuda=False)

    # compress the video and log the raw images before encoding
    writer = SummaryWriter(f"runs/{args.output}")
    getattr(compressor, args.compressor)(mask, args, logger, writer)    

     

if __name__ == "__main__":
    
    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        datefmt="%H:%M:%S",
        level="INFO",
    )

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-i',
        '--input',
        help='The input video file name. Must attach a corresponding mask file for compression purpose.',
        type=str,
        required=True
    )
    parser.add_argument(
        '-o',
        '--output',
        help='The output mp4 file name. Will attach a args file that contain args for decoding purpose.',
        type=str,
        required=True
    )
    parser.add_argument(
        '-p',
        '--pad',
        help='The padding size that pads extra high quality regions around existing high quality regions',
        type=int,
        required=True
    )
    parser.add_argument(
        '-c',
        '--compressor',
        help='The compressor used to compress the video.',
        type=str,
        required=True
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        help="The original video source.",
        required=True,
    )
    parser.add_argument(
        "--tile_size", type=int, help="The tile size of the mask.", default=16
    )
    parser.add_argument(
        '--preserve',
        help='Preserve source png folders for debugging purpose.',
        action='store_true'
    )
    parser.add_argument(
        "--smooth_frames",
        type=int,
        help="Proposing one single mask for smooth_frames many frames",
        default=30,
    )
    parser.add_argument(
        "--visualize_step_size",
        type=int,
        help="Proposing one single mask for smooth_frames many frames",
        default=100,
    )
    parser.add_argument("--qp", type=int, required=True)

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--bound", type=float, help="The lower bound for the mask. Exclusive with --perc",
    )
    action.add_argument(
        "--perc", type=float, help="The percentage of pixels in high quality. Exclusive with --bound"
    )
    args = parser.parse_args()
    
    main(args)