import os
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from PIL import Image

folder = Path("./bus")

gamma = 0.05

# def process(pair):
#     fid, name = pair
#     image = Image.open(name).resize((1280, 720))

#     output_folder = Path("./bear")

#     image.save(output_folder / ("%010d.png" % (fid)))


def process1(pair):
    fid, name = pair
    image = Image.open(name).resize((1280, 720))

    width, height = image.size

    l, r = 0, width
    t, b = 0, height * gamma

    # r = r * 4 // 5
    # l = r * 4 // 5 + 1

    image = image.crop((l, t, r, b))

    new_image = Image.new("RGB", (1280, 720))
    new_image.paste(image)

    output_folder = Path("./bus_crop")

    new_image.save(output_folder / ("%010d.png" % (fid)))


def process2(pair):
    fid, name = pair
    image = Image.open(name).resize((1280, 720))

    width, height = image.size

    l, r = 0, width
    t, b = 0, height * gamma

    r = r * 4 // 5
    # l = r * 4 // 5 + 1

    image = image.crop((l, t, r, b))

    new_image = Image.new("RGB", (1280, 720))
    new_image.paste(image)

    output_folder = Path("./bus_hq")

    new_image.save(output_folder / ("%010d.png" % (fid)))


def process3(pair):
    fid, name = pair
    image = Image.open(name).resize((1280, 720))

    width, height = image.size

    l, r = 0, width
    t, b = 0, height * gamma

    # r = r * 4 // 5
    l = r * 4 // 5 + 1

    image = image.crop((l, t, r, b))

    new_image = Image.new("RGB", (1280, 720))
    new_image.paste(image)

    output_folder = Path("./bus_lq")

    new_image.save(output_folder / ("%010d.png" % (fid)))


Parallel(n_jobs=32)(
    delayed(process1)(pair)
    for pair in enumerate(sorted(list(folder.glob("*.png"))))
)

Parallel(n_jobs=32)(
    delayed(process2)(pair)
    for pair in enumerate(sorted(list(folder.glob("*.png"))))
)

Parallel(n_jobs=32)(
    delayed(process3)(pair)
    for pair in enumerate(sorted(list(folder.glob("*.png"))))
)

os.system("rm bus_lq.mp4 bus_hq.mp4 bus_crop.mp4")

os.system("ffmpeg -i bus_lq/%010d.png -qp 36 bus_lq.mp4")

os.system("ffmpeg -i bus_hq/%010d.png -qp 24 bus_hq.mp4")

os.system("ffmpeg -i bus_crop/%010d.png -qp 24 bus_crop.mp4")

import time

time.sleep(2)

os.system("ls -sh bus_crop.mp4 bus_hq.mp4 bus_lq.mp4")
