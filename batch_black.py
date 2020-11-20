
import os
from itertools import product
import subprocess


# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ['train_first/trafficcam_%d_train' % (i+1) for i in range(4)] + ['train_first/dashcam_%d_train' % (i+1) for i in range(4)]
# v_list = [v_list[4]]

v_list = ['train_first/trafficcam_1_train']
tile = 16
# perc = 40
niter = 14
delta = 16

for v in v_list:

    output = f'{v}_compressed_black.hevc'

    subprocess.run([
        'python', 'compress_black.py',
        '-i', f'youtube_videos/{v}_qp_22.hevc',
        '-g', f'youtube_videos/{v}_qp_22.hevc',
        '-s', f'youtube_videos/{v}.yuv',
        '-o', f'youtube_videos/{output}',
        # '--tile_percentage', f'{perc}', 
        '--num_iterations', f'{niter}',
        '--tile_size', f'{tile}',
        '--delta', f'{delta}'
    ])
    # os.system(f'rm youtube_videos/{output}.qp{base}')
    os.system(f'python inference.py -i youtube_videos/{output}')
    os.system(f'python examine.py -i youtube_videos/{output} -g youtube_videos/{v}_qp_22.hevc')

