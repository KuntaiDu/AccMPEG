
import os

names = ['dashcamcropped_%d' % i for i in range(1, 2)]

for name in names:
    os.system(f'mkdir {name}')
    os.system(f'ffmpeg -i {name}.mp4 -start_number 0 {name}/%010d.png')
