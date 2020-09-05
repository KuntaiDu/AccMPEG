
import torch
from .video_utils import read_video, write_video
from pathlib import Path
import os

def generate_masked_image(mask, video_slices, bws):

    masked_image = torch.zeros_like(mask)

    for i in range(len(video_slices) - 1):

        x0, x1 = bws[i], bws[i+1]
        y0, y1 = video_slices[i], video_slices[i+1]

        if x1 == 1:
            inds = torch.logical_and(x0 <= mask, mask <= x1)
        else:
            inds = torch.logical_and(x0 <= mask, mask < x1)

        term = y0 + (mask - x0) / (x1 - x0) * (y1 - y0)
        masked_image += torch.where(
            inds,
            term,
            torch.zeros_like(mask)
        )

    return masked_image

def tile_mask(mask, tile_size):
    '''
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
    '''
    mask = mask[0, 0, :, :]
    t = tile_size
    mask = mask.unsqueeze(1).repeat(1, t, 1).view(-1, mask.shape[1])
    mask = mask.transpose(0, 1)
    mask = mask.unsqueeze(1).repeat(1, t, 1).view(-1, mask.shape[1])
    mask = mask.transpose(0, 1)
    return torch.cat(3 * [mask[None, None, :, :]], 1)

def tile_masks(mask, tile_size):
    '''
        Here the mask is of shape [N, 1, H, W]
    '''

    return torch.cat([tile_mask(mask_slice, tile_size) for mask_slice in mask.split(1)])

def mask_clip(mask, minval):
    mask.requires_grad = False
    mask[mask<minval] = minval
    mask[mask>1] = 1
    mask.requires_grad = True

def binarize_mask(mask, bw):
    assert sorted(bw) == bw
    assert mask.requires_grad == False

    for i in range(len(bw) - 1):

        mid = (bw[i] + bw[i+1]) / 2

        mask[torch.logical_and(mask > bw[i], mask <= mid)] = bw[i]
        mask[torch.logical_and(mask > mid, mask < bw[i+1])] = bw[i+1]

def generate_masked_video(mask, videos, bws, args):
    
    masked_video = torch.zeros_like(videos[-1])
    
    for fid, (video_slices, mask_slice) in enumerate(zip(zip(*[video.split(1) for video in videos]), mask.split(1))):
        mask_slice = tile_mask(mask_slice, args.tile_size)
        masked_image = generate_masked_image(mask_slice, video_slices, bws)
        masked_video[fid:fid+1, :, :, :] = masked_image

    return masked_video

def write_masked_video(mask, args, qps, bws, logger):

    # write several temporal mp4 files, and then use ffmpeg to compress them

    logger.info('Read the source video, for compression purpose...')
    video = read_video(args.source, logger)
    os.system(f'rm {args.output}*')

    for i, bw in enumerate(bws):

        qp = qps[i]

        if qp == -1:
            continue

        print(qp)

        bw_mask = torch.where(mask == bw, torch.ones_like(mask), torch.zeros_like(mask))
        write_video(tile_masks(bw_mask, args.tile_size) * video, f'temp_{qp}_uncompressed.mp4', logger)
        os.system(f'ffmpeg -y -i temp_{qp}_uncompressed.mp4 -c:v libx264 -qmin {qp} -qmax {qp} {args.output}')
        os.system(f'mv {args.output} {args.output}.qp{qp}')
        os.system(f'rm temp_{qp}_uncompressed.mp4')
        
        
        
        