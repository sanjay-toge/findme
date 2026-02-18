import yt_dlp
import os

VIDEOS = [
    "https://youtu.be/9a4izd3Rvdw?list=RD9a4izd3Rvdw",
]

def download_video(url):
    ydl_opts = {
        "outtmpl": "videos/%(id)s.%(ext)s",
        "format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    os.makedirs("videos", exist_ok=True)

    for url in VIDEOS:
        download_video(url)

    print("Videos downloaded 🎉")
