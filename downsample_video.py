import shutil
from pathlib import Path

video_list = ["dashcam/dashcamcropped_%d" % i for i in range(1, 10)]

# 30x downsample
for src in video_list:

    idx = 0

    x = Path(src)
    y = Path(src + "_downsampled")

    print(src)

    if not y.exists():
        y.mkdir()

    while True:

        if not (x / ("%010d.png" % idx)).exists():
            break

        shutil.copy(
            str(x / ("%010d.png" % idx)), str(y / ("%010d.png" % (idx // 30)))
        )

        idx += 30

        if idx % 300 == 0:
            print(idx)
