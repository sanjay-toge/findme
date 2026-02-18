import faiss
import numpy as np
import pickle
import os

INDEX_FILE = "stored_embeddings/faiss.index"
META_FILE = "stored_embeddings/meta.pkl"

dimension = 512

# 🔥 IMPORTANT: cosine similarity index
index = faiss.IndexFlatIP(dimension)

metadata = []

def load_db():
    global index, metadata

    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        metadata = pickle.load(open(META_FILE, "rb"))
    else:
        print("Creating new FAISS index")

def save_db():
    faiss.write_index(index, INDEX_FILE)
    pickle.dump(metadata, open(META_FILE, "wb"))

def normalize(vec):
    vec = np.array(vec).astype("float32")
    faiss.normalize_L2(vec)
    return vec

def add_embedding(embedding, data):
    vec = normalize([embedding])
    index.add(vec)
    metadata.append(data)
    # save_db()

def search_embedding(embedding, k=5):
    vec = normalize([embedding])

    distances, indices = index.search(vec, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        results.append({
            "score": float(distances[0][i]),
            "data": metadata[idx]
        })

    return results

load_db()


def get_total_faces():
    """Return the total number of embeddings stored in the FAISS index.

    Uses the FAISS index `ntotal` property when available, falling back to
    the length of the `metadata` list.
    """
    try:
        return int(index.ntotal)
    except Exception:
        return len(metadata)
