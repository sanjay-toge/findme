import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vector_db import save_db
import os
from deepface import DeepFace
from vector_db import add_embedding

FRAMES_DIR = "frames"

def get_timestamp(frame_name):
    # frame_0001.jpg → second 1
    return int(frame_name.split("_")[1].split(".")[0])

def process_frame(frame_path, video_id, frame_name):
    try:
        faces = DeepFace.represent(
            img_path=frame_path,
            model_name="ArcFace",
            detector_backend="mtcnn",
            enforce_detection=False
        )

        for face in faces:
            embedding = face["embedding"]

            metadata = {
                "video_id": video_id,
                "timestamp": get_timestamp(frame_name),
                "frame_path": frame_path
            }

            add_embedding(embedding, metadata)

    except Exception as e:
        print("Error:", e)

def process_all_frames():
    for video_folder in os.listdir(FRAMES_DIR):
        video_path = os.path.join(FRAMES_DIR, video_folder)

        if not os.path.isdir(video_path):
            continue

        print("Processing video:", video_folder)
        save_db()
        for frame in os.listdir(video_path):
            frame_path = os.path.join(video_path, frame)
            process_frame(frame_path, video_folder, frame)

if __name__ == "__main__":
    process_all_frames()
    print("All frames processed 🎉")
