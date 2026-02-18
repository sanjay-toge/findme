from fastapi import FastAPI, UploadFile, File
from face_engine import generate_embedding, compare_embeddings
from vector_db import add_embedding, search_embedding, get_total_faces
from deepface import DeepFace

app = FastAPI()
MATCH_THRESHOLD = 0.45   # lower recall threshold
MAX_CANDIDATES = 20     # number of faces to verify
VERIFY_DISTANCE_THRESHOLD = 0.68
STRICT_MATCH_THRESHOLD = 0.60

@app.post("/embedding")
async def create_embedding(file: UploadFile = File(...)):
    image_bytes = await file.read()
    embedding = generate_embedding(image_bytes)
    return {"embedding": embedding}

@app.post("/compare")
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    emb1 = generate_embedding(await file1.read())
    emb2 = generate_embedding(await file2.read())

    similarity = compare_embeddings(emb1, emb2)
    return {"similarity": similarity}

@app.post("/add-face")
async def add_face(file: UploadFile = File(...), label: str = "unknown"):
    emb = generate_embedding(await file.read())
    add_embedding(emb, {"label": label})
    return {"status": "stored"}


@app.get("/faces/count")
async def faces_count():
    """Return the total number of stored face embeddings."""
    total = get_total_faces()
    return {"total": total}

# @app.post("/search")
# async def search_face(file: UploadFile = File(...)):
#     emb = generate_embedding(await file.read())
#     results = search_embedding(emb)
#     return {"matches": results}

# @app.post("/search")
# async def search_face(file: UploadFile = File(...)):
#     emb = generate_embedding(await file.read())
#     matches = search_embedding(emb, k=50)  # get many matches

#     formatted = format_results(matches)

#     return {"results": formatted}

# @app.post("/search")
# async def search_face(file: UploadFile = File(...)):
#     emb = generate_embedding(await file.read())
#     matches = search_embedding(emb, k=50)

#     # 🔥 FILTER LOW CONFIDENCE MATCHES
#     strong_matches = [m for m in matches if m["score"] >= MATCH_THRESHOLD]

#     if len(strong_matches) == 0:
#         return {"results": []}

#     formatted = format_results(strong_matches)
#     return {"results": formatted}
# MATCH_THRESHOLD = 0.50

# @app.post("/search")
# async def search_face(file: UploadFile = File(...)):
#     emb = generate_embedding(await file.read())
#     matches = search_embedding(emb, k=50)

#     # 1️⃣ keep only strong matches
#     strong_matches = [m for m in matches if m["score"] >= MATCH_THRESHOLD]

#     # 2️⃣ keep only video embeddings (skip old test faces)
#     video_matches = [
#         m for m in strong_matches
#         if "video_id" in m["data"]
#     ]

#     if len(video_matches) == 0:
#         return {"results": []}

#     formatted = format_results(video_matches)
#     return {"results": formatted}

# @app.post("/search")
# async def search_face(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     emb = generate_embedding(image_bytes)

#     # Step 1: vector search (broad search)
#     matches = search_embedding(emb, k=MAX_CANDIDATES)

#     # Step 2: keep only strong candidates
#     candidates = [
#         m for m in matches
#         if m["score"] >= MATCH_THRESHOLD and "video_id" in m["data"]
#     ]

#     # Step 3: verify identity (strict check)
#     # verified_matches = []

#     # for match in candidates:
#     #     frame_path = match["data"]["frame_path"]

#     #     is_verified = verify_faces(image_bytes, frame_path)

#     #     if is_verified:
#     #         verified_matches.append(match)

#     verified_matches = []

#     for match in candidates:
#         frame_path = match["data"]["frame_path"]

#         distance = verify_face_distance(image_bytes, frame_path)

#         if distance <= VERIFY_DISTANCE_THRESHOLD:
#             verified_matches.append(match)


#     if len(verified_matches) == 0:
#         return {"results": []}

#     formatted = format_results(verified_matches)
#     return {"results": formatted}

@app.post("/search")
async def search_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    emb = generate_embedding(image_bytes)

    # Step 1 — broad vector search
    matches = search_embedding(emb, k=50)

    # Step 2 — keep only video embeddings
    video_matches = [
        m for m in matches
        if "video_id" in m["data"]
    ]

    # Step 3 — apply TWO thresholds
    strong_matches = [
        m for m in video_matches
        if m["score"] >= MATCH_THRESHOLD      # recall filter (0.45)
        and m["score"] >= STRICT_MATCH_THRESHOLD  # identity filter (0.60)
    ]

    if len(strong_matches) == 0:
        return {"results": []}

    formatted = format_results(strong_matches)
    return {"results": formatted}


def seconds_to_hms(seconds):
    m = seconds // 60
    s = seconds % 60
    return f"{m}:{s:02d}"

def youtube_link(video_id, timestamp):
    return f"https://www.youtube.com/watch?v={video_id}&t={timestamp}s"

def cluster_timestamps(timestamps, gap=10):
    timestamps = sorted(timestamps)
    clusters = []
    current = [timestamps[0]]

    for t in timestamps[1:]:
        if t - current[-1] <= gap:
            current.append(t)
        else:
            clusters.append(current)
            current = [t]

    clusters.append(current)
    return [int(sum(c)/len(c)) for c in clusters]

def format_results(matches):
    videos = {}

    for match in matches:
        video_id = match["data"]["video_id"]
        timestamp = match["data"]["timestamp"]

        if video_id not in videos:
            videos[video_id] = []

        videos[video_id].append(timestamp)

    final_results = []

    for video_id, timestamps in videos.items():
        scene_times = cluster_timestamps(timestamps)

        for t in scene_times:
            final_results.append({
                "video_id": video_id,
                "time_seconds": t,
                "time_human": seconds_to_hms(t),
                "youtube_url": youtube_link(video_id, t)
            })

    return final_results

# def verify_faces(uploaded_image_bytes, frame_path):
#     try:
#         result = DeepFace.verify(
#             img1_path = uploaded_image_bytes,
#             img2_path = frame_path,
#             model_name = "ArcFace",
#             detector_backend = "mtcnn",
#             enforce_detection=False
#         )
#         return result["verified"]
#     except:
#         return False

def verify_face_distance(uploaded_image_bytes, frame_path):
    try:
        result = DeepFace.verify(
            img1_path = uploaded_image_bytes,
            img2_path = frame_path,
            model_name = "ArcFace",
            detector_backend = "mtcnn",
            enforce_detection=False
        )
        return result["distance"]
    except:
        return 1.0  # very different
