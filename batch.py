
import os
from itertools import product

mask_weight_list = [400]
mask_p_list = [1]
cont_weight_list = [0]
cont_p_list = [1]
base_list = [38]
tile_list = [8]
v_list = ['trafficcam_1_short']

for mask_weight, mask_p, cont_weight, cont_p, base, tile, v in product(mask_weight_list, mask_p_list, cont_weight_list, cont_p_list, base_list, tile_list, v_list):
    
    output = f'{v}_compressed_loss.mp4'

    os.system(f'python compress.py -i youtube_videos/test/{v}_qp_{base}.mp4 youtube_videos/test/{v}_qp_24.mp4 -s youtube_videos/test/{v}.mp4 -o youtube_videos/test/{output} --mask_weight {mask_weight} --mask_p {mask_p} --cont_weight {cont_weight} --cont_p {cont_p} --learning_rate 0.005 --num_iterations 200 --tile_size {tile} -g youtube_videos/test/{v}_qp_24.mp4')
    os.system(f'rm youtube_videos/test/{output}.qp{base}')
    os.system(f'cp youtube_videos/test/{v}_qp_{base}.mp4 youtube_videos/test/{output}.qp{base}')
    os.system(f'python inference.py -i youtube_videos/test/{output}')
    os.system(f'python examine.py -i youtube_videos/test/{output} -g {v}_qp_24.mp4')