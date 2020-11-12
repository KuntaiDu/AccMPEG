
import os
from itertools import product
import yaml


# v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
# v_list = [v_list[0]]

v_list = ['train_first/dashcam_1_train', 'cross/dashcam_1_cross', 'test/dashcam_1_test']
# v_list = [v_list[2]]
base = 34
tile = 16
perc = 20
model = 'youtube_tile_16_2%_small_model_cross'

for v in v_list:

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    output = f'{v}_compressed_{model}.mp4'
    gt = f'{v}_compressed_gt.mp4'

    os.system(f'python compress_maskgen.py -i youtube_videos/{v}_qp_{base}.mp4 '
              f' youtube_videos/{v}_qp_24.mp4 -s youtube_videos/{v} -o youtube_videos/{output} --tile_size {tile}  -p maskgen_pths/fcn_mask_{model}.pth.best'
              f' --tile_percentage {perc} --mask youtube_videos/{gt}')
    # os.system(f'rm youtube_videos/{output}.qp{base}')
    os.system(f'cp youtube_videos/{v}_qp_{base}.mp4 youtube_videos/{output}.qp{base}')
    os.system(f'python inference.py -i youtube_videos/{output}')
    os.system(f'python examine.py -i youtube_videos/{output} -g youtube_videos/{v}_qp_24.mp4')
    with open('temp.txt', 'r') as f:
        acc = float(f.read())

    with open('stats', 'r') as f:
        data_full = yaml.load(f.read())
    if data_full[-1]['video_name'] == f'youtube_videos/{output}':
        data_full[-1]['f1'] = acc
    with open('stats', 'w') as f:
        f.write(yaml.dump(data_full))

