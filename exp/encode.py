import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torchvision.transforms as T
from PIL import Image

src = "../large_dashcam/large_dashcam_1/%010d.png"

x1, y1 = 400, 0
x2, y2 = 400, 300
delta = 300

with ThreadPoolExecutor(max_workers=4) as executor:
    for i in range(200):

        if i % 10 == 0:
            print(i)

        x = T.ToTensor()(Image.open(src % i))

        maskA = torch.zeros_like(x)
        maskA[:, x1 : x1 + delta, y1 : y1 + delta] = 1

        maskB = torch.zeros_like(x)
        maskB[:, x2 : x2 + delta, y2 : y2 + delta] = 1

        executor.submit(T.ToPILImage()(x * maskA).save, "./A2/%010d.png" % i)
        executor.submit(T.ToPILImage()(x * maskB).save, "./B2/%010d.png" % i)
        executor.submit(T.ToPILImage()(x * (maskA + maskB)).save, "./C2/%010d.png" % i)

os.system("ffmpeg -i A2/%010d.png -y -start_number 0 -qp 30 A2.mp4")
os.system("ffmpeg -i B2/%010d.png -y -start_number 0 -qp 40 B2.mp4")
os.system("ffmpeg -i C2/%010d.png -y -start_number 0 -qp 30 C2.mp4")
os.system("ffmpeg -i A/%010d.png -y -start_number 0 -qp 30 A.mp4")
os.system("ffmpeg -i B/%010d.png -y -start_number 0 -qp 40 B.mp4")
os.system("ffmpeg -i C/%010d.png -y -start_number 0 -qp 30 C.mp4")
