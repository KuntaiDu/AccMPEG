names = ["vis_%d" % i for i in range(169, 174)]
attrs = ["C4", "FPN", "DC5"]

import subprocess
from itertools import product

# for name, attr in product(names, attrs):

#     app = f"COCO-Detection/faster_rcnn_R_101_{attr}_3x.yaml"

#     subprocess.run(
#         [
#             "python",
#             "examine_reducto.py",
#             "--input",
#             f"visdrone/videos/{name}_reducto_qp_28_{attr}.mp4",
#             "--source",
#             f"visdrone/videos/{name}",
#             "-g",
#             f"visdrone/videos/{name}_qp_30.mp4",
#             "--stats",
#             f"stats_{attr}",
#             "--json",
#             f"baselines/jsons/{name}.json",
#             "--app",
#             app,
#         ]
#     )

names = ["dashcam_%d" % i for i in range(1, 11)]
attrs = ["C4", "FPN", "DC5"]

for name, attr in product(names, attrs):

    app = f"COCO-Detection/faster_rcnn_R_101_{attr}_3x.yaml"

    subprocess.run(
        [
            "python",
            "examine_reducto.py",
            "--input",
            f"dashcam/{name}_reducto_qp_28_{attr}.mp4",
            "--source",
            f"dashcam/{name}",
            "-g",
            f"dashcam/{name}_qp_30.mp4",
            "--stats",
            f"stats_{attr}",
            "--json",
            f"baselines/jsons/{name}.json",
            "--app",
            app,
        ]
    )
