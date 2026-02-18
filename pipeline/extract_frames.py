import os
import subprocess

VIDEOS_DIR = "videos"
FRAMES_DIR = "frames"

def extract_frames(video_file):
    video_id = os.path.splitext(video_file)[0]
    output_folder = f"{FRAMES_DIR}/{video_id}"
    os.makedirs(output_folder, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", f"{VIDEOS_DIR}/{video_file}",
        "-vf", "fps=1",
        f"{output_folder}/frame_%04d.jpg"
    ]

    subprocess.run(cmd)

if __name__ == "__main__":
    os.makedirs(FRAMES_DIR, exist_ok=True)

    for video in os.listdir(VIDEOS_DIR):
        extract_frames(video)

    print("Frames extracted 🎉")
