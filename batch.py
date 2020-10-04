
import os
from itertools import product

mask_weight_list = [400]
mask_p_list = [1]
cont_weight_list = [0]
cont_p_list = [1]
base_list = [38]
tile_list = [8]
v_list = ['trafficcam_2', 'trafficcam_3', 'trafficcam_4', 'dashcam_1']

for mask_weight, mask_p, cont_weight, cont_p, base, tile, v in product(mask_weight_list, mask_p_list, cont_weight_list, cont_p_list, base_list, tile_list, v_list):
    
    output = f'{v}_loss_compressed_mask_weight_{mask_weight}_mask_p_{mask_p}_cont_weight_{cont_weight}_cont_p_{cont_p}_base_{base}_tile_{tile}.mp4'

    os.system(f'python compress.py -i videos/{v}_qp_{base}.mp4 videos/{v}_qp_24.mp4 -s videos/{v}.mp4 -o videos/{output} --mask_weight {mask_weight} --mask_p {mask_p} --cont_weight {cont_weight} --cont_p {cont_p} --learning_rate 0.005 --num_iterations 200 --tile_size {tile} -g videos/{v}_qp_24.mp4')
    os.system(f'rm videos/{output}.qp{base}')
    os.system(f'cp videos/{v}_qp_{base}.mp4 videos/{output}.qp{base}')
    os.system(f'python inference.py -i videos/{output}')
    os.system(f'python examine.py -i videos/{output} -g {v}_qp_24.mp4')