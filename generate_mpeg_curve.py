
import os
import argparse
from pathlib import Path
import subprocess
import logging
from utils.video_utils import encode_with_qp

qp_list = [22, 23, 24, 26, 28, 32, 34, 38]
# qp_list = [qp_list[0]]


def main(args):

    for video_name in args.inputs:
        assert Path(
            video_name).suffix == '.yuv', 'The base video should be a yuv file'
        video_name = Path(video_name)
        video_name = video_name.parent / video_name.stem

        for qp in qp_list:
            input_name = f'{video_name}.yuv'
            output_name = f'{video_name}_qp_{qp}.hevc'
            print(f'Generate video for {output_name}')
            encode_with_qp(input_name, output_name, qp, args)

            subprocess.run([
                'python',
                'inference.py',
                '-i', output_name
            ])
            subprocess.run([
                'python',
                'examine.py',
                '-i', output_name,
                '-g', f'{video_name}_qp_22.hevc'
            ])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputs', nargs='+',
                        help='The video file names. The largest video file will be the ground truth.', required=True)
    parser.add_argument('--tile_size', type=int,
                        help='The tile size of the mask.', default=16)
    args = parser.parse_args()
    main(args)
