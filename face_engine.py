from deepface import DeepFace
import numpy as np
import cv2
from PIL import Image
import io

MODEL_NAME = "ArcFace"

def image_bytes_to_array(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)

def generate_embedding(image_bytes):
    img = image_bytes_to_array(image_bytes)

    embedding = DeepFace.represent(
        img_path = img,
        model_name = MODEL_NAME,
        enforce_detection=True
    )

    return embedding[0]["embedding"]

def compare_embeddings(emb1, emb2):
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(similarity)
