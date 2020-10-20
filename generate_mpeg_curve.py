
import os
import argparse
from pathlib import Path
import subprocess

qp_list = [24, 38, 25, 26, 28, 30, 34]

def main(args):

    for video_name in args.inputs:
        # video_name should be a png directory
        assert Path(video_name).is_dir()

        for qp in qp_list:
            output_name = f'{video_name}_qp_{qp}.mp4'
            subprocess.run([
                'ffmpeg',
                '-y',
                '-i', video_name + '/%05d.png',
                '-start_number', '0',
                '-c:v', 'libx264',
                '-qmin', f'{qp}',
                '-qmax', f'{qp}',
                output_name
            ])
            subprocess.run([
                'python',
                'inference.py',
                '-i', output_name
            ])
            subprocess.run([
                'python',
                'examine.py',
                '-i', output_name,
                '-g', f'{video_name}_qp_24.mp4'
            ])

        #os.system(f'python examine.py -i {video_names} -g {video_name}_qp_24.mp4')
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputs', nargs = '+', help='The video file names. The largest video file will be the ground truth.', required=True)
    args = parser.parse_args()
    main(args)