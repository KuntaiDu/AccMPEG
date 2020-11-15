
import torch

from . import video_utils as vu
from . import bbox_utils as bu
from pathlib import Path
import os
import glob
import pickle
import subprocess
import struct

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
    # assert mask.requires_grad == False

    mask_ret = mask.detach().clone()

    for i in range(len(bw) - 1):

        mid = (bw[i] + bw[i+1]) / 2

        mask_ret[torch.logical_and(mask > bw[i], mask <= mid)] = bw[i]
        mask_ret[torch.logical_and(mask > mid, mask < bw[i+1])] = bw[i+1]
    return mask_ret

def generate_masked_video(mask, videos, bws, args):
    
    masked_video = torch.zeros_like(videos[-1])
    
    for fid, (video_slices, mask_slice) in enumerate(zip(zip(*[video.split(1) for video in videos]), mask.split(1))):
        mask_slice = tile_mask(mask_slice, args.tile_size)
        masked_image = generate_masked_image(mask_slice, video_slices, bws)
        masked_video[fid:fid+1, :, :, :] = masked_image

    return masked_video

def encode_masked_video(args, qp, binary_mask, logger):

    # create temp folder to write png files
    from datetime import datetime
    temp_folder = 'temp_' + datetime.now().strftime('%m.%d.%Y,%H:%M:%S')

    Path(temp_folder).mkdir()

    from PIL import Image
    from torchvision import transforms as T
    import enlighten

    # make a progress bar
    progress_bar = enlighten.get_manager().counter(total=binary_mask.shape[0], desc=f'Generate raw png of {args.output}.qp{qp}', unit='frames')

    subprocess.run([
        'mkdir', args.source + '.pngs'
    ])

    subprocess.run([
        'ffmpeg',
        '-y', 
        '-f', 'rawvideo',
        '-pix_fmt', 'yuv420p',
        '-s:v', '1280x720',
        '-i', args.source
        args.source + '.pngs/%010d.png'
    ])

    # read png files from source and multiply it by mask
    for i in range(binary_mask.shape[0]):
        progress_bar.update()
        source_image = Image.open(args.source + '.pngs/%010d.png' % i)
        image_tensor = T.ToTensor()(source_image)
        binary_mask_slice = tile_mask(binary_mask[i:i+1, :, :, :], args.tile_size)
        image_tensor = image_tensor * binary_mask_slice
        dest_image = T.ToPILImage()(image_tensor[0, :, :, :])
        dest_image.save(temp_folder + '/%010d.png' % i)

    subprocess.run([
        'ffmpeg',
        '-i', temp_folder + '/%010d.png',
        '-start_number', '0',
        '-vcodec', 'rawvideo',
        '-an', 
        args.output + '.yuv'
    ])

    # encode it through ffmpeg
    subprocess.run([
        'kvazaar', 
        '--input', args.output + '.yuv',
        '--input-res', '1280x720',
        '-q', f'{qp}',
        '--gop', '0',
        '--output', args.output
    ])

    # annotate the video quality
    subprocess.run([
        'mv', args.output, f'{args.output}.qp{qp}'
    ])

    # import pdb; pdb.set_trace()

    # remove temp folder
    subprocess.run([
        'rm', '-r', temp_folder
    ])

    subprocess.run([
        'rm', args.output + '.yuv'
    ])

def write_masked_video(mask, args, qps, bws, logger):

    mask = mask[:, 0, :, :]
    mask = mask.permute(0, 2, 1)
    delta_qp0 = torch.ones_like(mask) * (qps[0] - 22)
    delta_qp1 = torch.ones_like(mask) * (qps[1] - 22)
    mask = torch.where(mask == 0, delta_qp0, delta_qp1).int()

    _, w, h = mask.shape

    with open('temp.dat', 'wb') as f:
        for fid in range(len(mask)):
            f.write(struct.pack('i', w))
            f.write(struct.pack('i', h))
            for j in range(h):
                for i in range(w):
                    f.write(struct.pack('b', mask[fid, i, j]))

    subprocess.run([
        'kvazaar',
        '--input', args.source,
        '--gop', '0',
        '--input-res', '1280x720',
        '--roi-file', 'temp.dat',
        '--output', args.output
    ])

    with open(f'{args.output}.args', 'wb') as f:
        pickle.dump(args, f)

def read_masked_video(video_name, logger):

    logger.info(f'Reading compressed video {video_name}. Reading each part...')
    parts = sorted(glob.glob(f'{video_name}.qp[0-9]*'), reverse=True)
    parts2 = []

    videos = []
    # import pdb; pdb.set_trace()
    # parts2 = ['youtube_videos/train_first/dashcam_1_train_qp_%d.mp4' % i for i in [24, 38]]
    for part in parts:
        videos.append(vu.read_video(part, logger))
    logger.info(f'Reading mask for compressed video.')

    with open(f'{video_name}.mask', 'rb') as f:
        filename2mask = pickle.load(f)
    tile_size = filename2mask['args.tile_size']

    base = videos[0]
    for video, part in zip(videos[1:], parts[1:]):
        video = video * tile_masks(filename2mask[part], tile_size)
        base[video != 0] = video[video != 0]
    return base

def generate_mask_from_regions(mask_slice, regions, minval):

    # (xmin, ymin, xmax, ymax)
    regions = bu.point_form(regions)
    mask_slice[:, :, :, :] = minval

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

    return mask_slice

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
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result