
import os
import subprocess
import itertools
import glob
from pathlib import Path
from datetime import datetime

video_list = ['dashcam_%d' % (i+1) for i in range(4)] + ['trafficcam_%d' % (i+1) for i in range(4)]

qp_list = [24, 34]

temp_folder = 'temp_' + datetime.now().strftime('%m.%d.%Y,%H:%M:%S')
Path(temp_folder).mkdir()


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
            '-start_number', '0',
            temp_folder + '/%05d.png'
        ])

        # encode to video

        subprocess.check_output([
            'ffmpeg',
            '-y',
            '-i', temp_folder + '/%05d.png',
            '-start_number', '0',
            '-c:v', 'libx264', '-qmin', f'{qp}', '-qmax', f'{qp}',
            f'{vpath}_qp_{qp}.mp4'
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
            '-i', f'{vpath}_qp_{qp}.mp4',
            '-start_number', f'{start_number+1}',
            f'{png_path}/%05d.png'
        ])

        # remove the encoded video
        os.system(f'rm {vpath}_qp_{qp}.mp4')

        # clear temp folder
        subprocess.run([
            'rm',
            temp_folder + '/*'
        ])


    # # test set
    # vpath = f'youtube_videos/{video}.mp4'

    # opath = f'youtube_videos/train_first/{video}_train'
    # Path(opath).mkdir(exist_ok=True)
    # subprocess.run([
    #     'ffmpeg',
    #     '-y',
    #     '-i', vpath,
    #     '-ss', '0:0:14',
    #     '-t', '0:0:10',
    #     '-start_number', '0',
    #     opath + '/%05d.png'
    # ])

    # # different encoding
    # opath = f'youtube_videos/train_last/{video}_train'
    # Path(opath).mkdir(exist_ok=True)
    # subprocess.run([
    #     'ffmpeg',
    #     '-y',
    #     '-i', vpath,
    #     '-ss', '0:0:40',
    #     '-t', '0:0:10',
    #     '-start_number', '0',
    #     opath + '/%05d.png'
    # ])

    # opath = f'youtube_videos/cross/{video}_cross'
    # Path(opath).mkdir(exist_ok=True)
    # subprocess.run([
    #     'ffmpeg',
    #     '-y',
    #     '-i', vpath,
    #     '-ss', '0:1:05',
    #     '-t', '0:0:4',
    #     '-start_number', '0',
    #     opath + '/%05d.png'
    # ])


    # opath = f'youtube_videos/test/{video}_test'
    # Path(opath).mkdir(exist_ok=True)
    # subprocess.run([
    #     'ffmpeg',
    #     '-y',
    #     '-i', vpath,
    #     '-ss', '0:1:10',
    #     '-t', '0:0:5',
    #     '-start_number', '0',
    #     opath + '/%05d.png'
    # ])

# remove temp folder
subprocess.run([
        'rm',
        '-r',
        temp_folder
    ])