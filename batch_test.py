
import os
from itertools import product


v_list = ['dashcam_%d_test' % (i+1) for i in range(4)] + ['trafficcam_%d_test' % (i+1) for i in range(4)]
v_list = [v_list[0]]
base = 38
tile = 8

for v in v_list:

    output = f'{v}_compressed_maskgen_downsample.mp4'

    os.system(f'python compress_maskgen.py -i youtube_videos/test/{v}_qp_{base}.mp4 '
              f' youtube_videos/test/{v}_qp_24.mp4 -s youtube_videos/test/{v} -o youtube_videos/test/{output} --tile_size {tile}  -p maskgen_pths/fcn_mask_weight_30_1_video.pth')
    os.system(f'rm youtube_videos/test/{output}.qp{base}')
    os.system(f'cp youtube_videos/test/{v}_qp_{base}.mp4 youtube_videos/test/{output}.qp{base}')
    os.system(f'python inference.py -i youtube_videos/test/{output}')
    os.system(f'python examine.py -i youtube_videos/test/{output} -g youtube_videos/test/{v}_qp_24.mp4')
