
import os
import argparse

qp_list = [24, 25, 26, 28, 30, 34, 38, 40, 42, 46, 51]

def main(args):

    for video_name in args.inputs:
        video_names = ""
        # remove suffix of file name
        video_name = '.'.join(video_name.split('.')[:-1])
        for qp in qp_list:
            output_name = f'{video_name}_qp_{qp}.mp4'
            os.system(f'ffmpeg -i {video_name}.mp4 -y -c:v libx264 -qmin {qp} -qmax {qp} {output_name}')
            os.system(f'python inference.py -i {output_name}')
            video_names += output_name
            video_names += ' '

        os.system(f'python examine.py -i {video_names} -g {video_name}_qp_24.mp4')
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputs', nargs = '+', help='The video file names. The largest video file will be the ground truth.', required=True)
    args = parser.parse_args()
    main(args)