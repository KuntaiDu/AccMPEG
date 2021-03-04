from pathlib import Path

from joblib import Parallel, delayed
from PIL import Image

folder = Path("./drive/")


def process(pair):
    fid, name = pair
    image = Image.open(name).resize((1280, 720))
    if fid % 100 == 0:
        print(fid)

    output_folder = Path("./drive_%d" % (fid // 60))

    image.save(output_folder / ("%010d.png" % (fid % 60)))


for i in range(60):
    Path("./drive_%d" % i).mkdir(exist_ok=True)
    # os.path.mkdir("drive_%d" % i, exist_ok=True)


Parallel(n_jobs=32)(
    delayed(process)(pair)
    for pair in enumerate(sorted(list(folder.glob("*.png"))))
)

