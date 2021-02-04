names = ["vis_%d" % i for i in range(169, 174)]

import subprocess

for name in names:

    subprocess.run(
        [
            "python",
            "examine_reducto.py",
            "--input",
            f"visdrone/videos/{name}_reducto_qp_28.mp4",
            "--source",
            f"visdrone/videos/{name}",
            "-g",
            f"visdrone/videos/{name}_qp_30.mp4",
            "--stats",
            "stats_detection",
            "--json",
            f"jsons/{name}.json",
        ]
    )
