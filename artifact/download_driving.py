import subprocess

urls = [
    "https://www.youtube.com/watch?v=TE2tfavIo3E",
    "https://www.youtube.com/watch?v=b-WViLMs_4c",
    "https://www.youtube.com/watch?v=CIxwONzDNto",
    "https://www.youtube.com/watch?v=OOPWzhFnCZg",
    "https://www.youtube.com/watch?v=3ma3yXXc3V8",
]

# urls = [urls[0]]

fmt = "ffmpeg -y -i $(youtube-dl -f 22 --get-url %s) -ss 00:00:35 -t 00:01:00  -c:v copy %s.mp4"

# crop_fmt = 'ffmpeg -y -i %s.mp4 -filter:v "crop=1280:600:1:1,scale=1280:720" -max_muxing_queue_size 900 %s.mp4'

for idx, url in enumerate(urls):

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

    subprocess.run(fmt % (url, f"driving_{idx}"), shell=True)

    # subprocess.run(
    #     crop_fmt % (f"dashcam_{idx}", f"dashcamcropped_{idx}"), shell=True
    # )

    # subprocess.run(["mkdir", f"dashcamcropped_{idx+1}"])

    subprocess.run(["mkdir", f"driving_{idx}"])

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            f"driving_{idx}.mp4",
            "-start_number",
            "0",
            f"driving_{idx}/%010d.png",
        ]
    )
