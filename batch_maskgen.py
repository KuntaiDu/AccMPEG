
import os
from itertools import product


# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

v_list = ['train_first/trafficcam_1_train', 'train_first/dashcam_1_train']
# v_list = [v_list[2]]
base = 34
tile = 16
perc = 2.5

for v in v_list:

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f'{v}_compressed_maskgen_COCO.mp4'

    os.system(f'python compress_maskgen.py -i youtube_videos/{v}_qp_{base}.mp4 '
              f' youtube_videos/{v}_qp_24.mp4 -s youtube_videos/{v} -o youtube_videos/{output} --tile_size {tile}  -p maskgen_pths/fcn_mask_COCO_half_focal_loss.pth1'
              f' --tile_percentage {perc}')
    # os.system(f'rm youtube_videos/{output}.qp{base}')
    os.system(f'cp youtube_videos/{v}_qp_{base}.mp4 youtube_videos/{output}.qp{base}')
    os.system(f'python inference.py -i youtube_videos/{output}')
    os.system(f'python examine.py -i youtube_videos/{output} -g youtube_videos/{v}_qp_24.mp4')

