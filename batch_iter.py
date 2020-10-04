
import os
from itertools import product

base_list = [42, 51]
niter_list = [60]

for base in base_list:
    for niter in niter_list:
    
        output = f'trafficcam_compressed_iter_base_{base}_niter_{niter}.mp4'

        os.system(f'python compress_iter.py -i trafficcam_{base}.mp4 trafficcam_24.mp4 -s trafficcam.mp4 -o {output} -g trafficcam_24.mp4 --num_iteration {niter}')
        os.system(f'rm {output}.qp{base}')
        os.system(f'cp trafficcam_{base}.mp4 {output}.qp{base}')
        os.system(f'python inference.py -i {output}')
        os.system(f'python examine.py -i {output} -g trafficcam_24.mp4')

    