
import torch
from torchvision import io
import os

def read_video(video_list, logger):
    '''
        Read a list of video and return two lists. 
        One is the video tensors, the other is the bandwidths.
    '''
    video_list = [{'video': read_single_video(video_name, logger), 
                   'bandwidth': read_single_bandwidth(video_name)}
                  for video_name in video_list]
    video_list = sorted(video_list, key=lambda x: x['bandwidth'])

    # bandwidth normalization
    gt_bandwidth = video_list[-1]['bandwidth']
    for i in video_list:
        i['bandwidth'] /= gt_bandwidth

    return [i['video'] for i in video_list], [i['bandwidth'] for i in video_list]

def read_single_video(video_name, logger):
    logger.info(f'Reading {video_name}')
    if 'mp4' in video_name:
        return io.read_video(video_name, pts_unit='sec')[0].float().div(255).permute(0, 3, 1, 2)

def read_single_bandwidth(video_name):
    if 'mp4' in video_name:
        return os.path.getsize(video_name)

def save_single_video(video_tensor, video_name, logger):

    logger.info(f'Saving {video_name}')

    # lossless encode. Should be replaced
    io.write_video(video_name, video_tensor, fps=25, options={'crf': 0})