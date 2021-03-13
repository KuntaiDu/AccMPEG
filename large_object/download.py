import subprocess

urls = [
    "https://www.youtube.com/watch?v=Pc0MDeOCjJw",
    "https://www.youtube.com/watch?v=AQe9NPjxTJw",
    "https://www.youtube.com/watch?v=sWIVBkVym8g",
    "https://www.youtube.com/watch?v=R7p8yUmq7ls",
]

fmt = "ffmpeg -y -i $(youtube-dl -f 136 --get-url %s) -ss 00:00:15 -t 00:01:00 -c:v copy %s.mp4"

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
    #         f"large_{idx+1}.mp4",
    #     ]
    # )

    subprocess.run(fmt % (url, f"large_{idx+1}"), shell=True)

    subprocess.run(["mkdir", f"large_{idx+1}"])

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            f"large_{idx+1}.mp4",
            "-start_number",
            "0",
            f"large_{idx+1}/%010d.png",
        ]
    )
