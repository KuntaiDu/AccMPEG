
import os
from itertools import product

mask_weight_list = [128, 32]
mask_p_list = [1, 3]
cont_weight_list = [0, 128, 32]
cont_p_list = [1, 3]
base_list = [34, 42, 51]
tile_list = [16, 80]

for mask_weight, mask_p, cont_weight, cont_p, base, tile in zip(mask_weight_list, mask_p_list, cont_weight_list, cont_p_list, base_list, tile_list):
    
    output = f'trafficcam_compressed_mask_weight_{mask_weight}_mask_p_{mask_p}_cont_weight_{cont_weight}_cont_p_{cont_p}_base_{base}_tile_{tile}.mp4'

    os.system(f'CUDA_VISIBLE_DEVICES=3 python compress.py -i trafficcam_{base}.mp4 trafficcam_24.mp4 -s trafficcam.mp4 -o {output} --mask_weight {mask_weight} --mask_p {mask_p} --cont_weight {cont_weight} --cont_p {cont_p} --learning_rate 0.06 --num_iterations 40 --tile_size {tile}')
    os.system(f'rm {output}.qp{base}')
    os.system(f'cp trafficcam_{base}.mp4 {output}.qp{base}')
    os.system(f'CUDA_VISIBLE_DEVICES=3 python inference.py -i {output}')
    os.system(f'python examine.py -i {output} -g trafficcam_24.mp4')