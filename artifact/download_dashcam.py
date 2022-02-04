import subprocess

urls = [
    "https://www.youtube.com/watch?v=R-b6dReO4QAA",
    "https://www.youtube.com/watch?v=eoXguTDnnHM",
    "https://www.youtube.com/watch?v=JZqEHDqwk3g",
    "https://www.youtube.com/watch?v=eUGFPOj8Pxg",
    "https://www.youtube.com/watch?v=kR9hnh4Y5rQ",
    "https://www.youtube.com/watch?v=KwYlOVeUiXI",
    "https://www.youtube.com/watch?v=q0dVSXeq3S0",
    # "https://www.youtube.com/watch?v=N5sJZ-KMd3I",
    # "https://www.youtube.com/watch?v=chrc7oSgayo",
    # "https://www.youtube.com/watch?v=gHWG7dWQ7kQ",
]

fmt = "ffmpeg -y -i $(youtube-dl -f 22 --get-url %s) -ss 00:20:00 -t 00:01:00  -c:v copy %s.mp4"

crop_fmt = 'ffmpeg -y -i %s.mp4 -filter:v "crop=1280:600:1:1,scale=1280:720" -max_muxing_queue_size 900 %s.mp4'

for idx, url in enumerate(urls):

    # if idx != 7:
    #     continue
    # if idx != 4:
    #     continue

    # ffmpeg -i $(youtube-dl -f 22 --get-url https://www.youtube.com/watch?v=ZbZSe6N_BXs) \
    # -ss 00:00:10 -t 00:00:30 -c:v copy -c:a copy \
    # happy.mp4

    # subprocess.run(
    #     [
    #         "ffmpeg",
    #         "-i",
    #         f"$(youtube-dl -f 136 --get-url {url})",
    #         "-ss",
    #         "0:20:0",
    #         "-t",
    #         "0:1:0",
    #         "-c:v",
    #         "copy",
    #         f"dashcam_{idx+1}.mp4",
    #     ]
    # )

    subprocess.run(fmt % (url, f"dashcam_{idx+1}"), shell=True)

    subprocess.run(
        crop_fmt % (f"dashcam_{idx+1}", f"dashcamcropped_{idx+1}"), shell=True
    )

    subprocess.run(["mkdir", f"dashcamcropped_{idx+1}"])

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            f"dashcamcropped_{idx+1}.mp4",
            "-start_number",
            "0",
            f"dashcamcropped_{idx+1}/%010d.png",
        ]
    )
