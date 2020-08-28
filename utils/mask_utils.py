
import torch

def generate_masked_image(mask, video_slices, bws):

    masked_image = torch.zeros_like(mask)

    for i in range(len(video_slices) - 1):

        x0, x1 = bws[i], bws[i+1]
        y0, y1 = video_slices[i], video_slices[i+1]

        if i == 0:
            x0 = -0.1

        term = y0 + (mask - x0) / (x1 - x0) * (y1 - y0)
        masked_image += torch.where(
            torch.logical_and(x0 < mask, mask <= x1),
            term,
            torch.zeros_like(mask)
        )

    return masked_image

def tile_mask(mask, tile_size):
    '''
        Eg: 
        mask = [    1   2
                    3   4]
        tile_mask(mask, 2) will return
        ret =  [    1   1   2   2
                    1   1   2   2
                    3   3   4   4
                    3   3   4   4]
        The goal is to control the granularity of the mask.        
    '''
    mask = mask[0, 0, :, :]
    t = tile_size
    mask = mask.unsqueeze(1).repeat(1, t, 1).view(-1, mask.shape[1])
    mask = mask.transpose(0, 1)
    mask = mask.unsqueeze(1).repeat(1, t, 1).view(-1, mask.shape[1])
    mask = mask.transpose(0, 1)
    return torch.cat(3 * [mask[None, None, :, :]], 1)

def mask_clip(mask):
    mask.requires_grad = False
    mask[mask<0] = 0
    mask[mask>1] = 1
    mask.requires_grad = True