
import os
import subprocess
import itertools
import glob
from pathlib import Path
from datetime import datetime

video_list = ['dashcam_%d' % (i+1) for i in range(4)] + ['trafficcam_%d' % (i+1) for i in range(4)]

qp_list = [24, 34]

os.system('rm youtube_videos/train_pngs_qp_*/*.png')

for video in video_list:

    # training set
    for qp in qp_list:

        vpath = f'youtube_videos/{video}'
        png_path = f'youtube_videos/train_pngs_qp_{qp}/'

        # decode to pngs
        subprocess.check_output([
            'ffmpeg',
            '-y',
            '-i', f'{vpath}.mp4',
            '-ss', '0:0:14',
            '-t', '0:0:50',
            '-vcodec', 'rawvideo',
            '-an',
            f'{vpath}.yuv'
        ])

        # encode to video
        subprocess.check_output([
            'kvazaar',
            '--input', f'{vpath}.yuv',
            '--input-res', '1280x720',
            '-q', f'{qp}',
            '--gop', '0',
            '--output', f'{vpath}_qp_{qp}.hevc'
        ])

        # get current largest number of image
        start_number = max([-1] + [
            int(i.split('.')[-2].split('/')[-1]) for i in glob.glob(
                png_path + '/*.png'
            )
        ])

        # decode the video into pngs
        subprocess.check_output([
            'ffmpeg',
            '-i', f'{vpath}_qp_{qp}.hevc',
            '-start_number', f'{start_number+1}',
            f'{png_path}/%010d.png'
        ])

        os.system(f'rm {vpath}_qp_{qp}.hevc')


    # test set
    vpath = f'youtube_videos/{video}.mp4'
    opath = f'youtube_videos/train_first/{video}_train'
    Path(opath).mkdir(exist_ok=True, parents=True)
    subprocess.run([
        'ffmpeg',
        '-y',
        '-i', vpath,
        '-ss', '0:0:14',
        '-t', '0:0:10',
        '-vcodec', 'rawvideo',
        '-an',
        f'{opath}.yuv'
    ])

    # different encoding
    opath = f'youtube_videos/train_last/{video}_train'
    Path(opath).mkdir(exist_ok=True,  parents=True)
    subprocess.run([
        'ffmpeg',
        '-y',
        '-i', vpath,
        '-ss', '0:0:40',
        '-t', '0:0:10',
        '-vcodec', 'rawvideo',
        '-an',
        f'{opath}.yuv'
    ])

    # test set
    opath = f'youtube_videos/test/{video}_test'
    Path(opath).mkdir(exist_ok=True, parents=True)
    subprocess.run([
        'ffmpeg',
        '-y',
        '-i', vpath,
        '-ss', '0:1:10',
        '-t', '0:0:5',
        '-vcodec', 'rawvideo',
        '-an',
        f'{opath}.yuv'
    ])
