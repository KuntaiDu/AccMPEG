
import os
from itertools import product


# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

v_list = ['train_first/dashcam_1_train', 'train_last/dashcam_1_train', 'cross/dashcam_1_cross', 'test/dashcam_1_test']
v_list = [v_list[2]]
base = 38
tile = 8

for v in v_list:

    output = f'{v}_compressed_maskgen_dashcam.mp4'

    os.system(f'python compress_maskgen.py -i youtube_videos/{v}_qp_{base}.mp4 '
              f' youtube_videos/{v}_qp_24.mp4 -s youtube_videos/{v} -o youtube_videos/{output} --tile_size {tile}  -p maskgen_pths/fcn_mask_weight_30_dashcam_1.pth')
    # os.system(f'rm youtube_videos/{output}.qp{base}')
    os.system(f'cp youtube_videos/{v}_qp_{base}.mp4 youtube_videos/{output}.qp{base}')
    os.system(f'python inference.py -i youtube_videos/{output}')
    os.system(f'python examine.py -i youtube_videos/{output} -g youtube_videos/{v}_qp_24.mp4')

