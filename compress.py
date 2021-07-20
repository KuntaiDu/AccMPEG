
import pickle
import argparse

from matplotlib.pyplot import streamplot
import torch
from utils.mask_utils import percentile


def main(args):
    
    with open(f"{args.input}.mask", "wb") as f:
        mask = pickle.load(f)
        
    splitted_mask = mask.split(args.smooth_frames)
    
    # smooth the mask first. 
    # we only sample the mask from the first image of every video segment.
    for i in range(len(splitted_mask)):
        cur_slice = splitted_mask[i]
        if i < len(splitted_mask) - 1:
            next_slice = splitted_mask[i+1]
            cur_slice[:, :, :, :] = 0.5 * (cur_slice[0]  + next_slice[0]).unsqueeze(0)
        else:
            cur_slice[:, :, :, :] = 0.5 * (cur_slice[0] + cur_slice[-1]).unsqueeze(0)
            
    # Two types of knobs: heat value threshold and knobs
    assert args.bound != -1. or args.perc != -1.
    assert not (args.bound != -1. and args.perc != -1.)
    
    # Binarize the mask using args.perc or args.bound
    if args.bound != -1:
        mask = dilate_binarize(mask, args.bound, args.conv_size, cuda=False)
    else:
        assert args.perc != -1
        mask = dilate_binarize(mask, percentile(mask, args.perc), args.conv_size, cuda=False)

    exec(f'compressor = {args.compressor}')
    compressor(mask, args, logger)     

if __name__ == "__main__":
    
    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        level="INFO",
    )

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-i',
        '--inputs',
        help='The input video file name. Must attach a corresponding mask file for compression purpose.',
        type=str,
        required=True
    )
    
    parser.add_argument(
        '-o',
        '--outputs',
        help='The output mp4 file name. Will attach a args file that contain args for decoding purpose.',
        type=str,
        required=True
    )
    
    parser.add_argument(
        '-p',
        '--pad_size',
        help='The padding size',
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
        '--preserve',
        help='Preserve source png folders for debugging purpose.',
        action='store_true'
    )
    
    main(args)