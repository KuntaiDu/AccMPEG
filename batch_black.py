
import os
from itertools import product
import subprocess


# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

# v_list = ['train_first/trafficcam_%d_train' % (i+1) for i in range(4)] + ['train_first/dashcam_%d_train' % (i+1) for i in range(4)]
# v_list = [v_list[4]]

v_list = ['visdrone/videos/vis_172']
tile = 16
# perc = 40
niter = 15
delta = 32

for v in v_list:

    output = f'{v}_compressed_black.hevc'

    subprocess.run([
        'python', 'compress_black.py',
        '-i', f'{v}_qp_22.hevc',
        '-g', f'{v}_qp_22.hevc',
        '-s', f'{v}.yuv',
        '-o', f'{output}',
        # '--tile_percentage', f'{perc}', 
        '--num_iterations', f'{niter}',
        '--tile_size', f'{tile}',
        '--delta', f'{delta}'
    ])
    os.system(f'python inference.py -i {output}')
    os.system(f'python examine.py -i {output} -g {v}_qp_22.hevc')

