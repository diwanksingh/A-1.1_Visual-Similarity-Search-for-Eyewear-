import csv
import os
import faiss
import numpy as np
from collections import defaultdict

META_FILE = "data/metadata.csv"
INDEX_FILE = "index.faiss"
CENTROID_FILE = "centroids.npy"
# CLIP embedding dimension (fixed)
DIM = 512
class VectorStore:
    """
    FAISS based vector store for:
    - similarity search
    - rough style classification
    """
    def __init__(self):
        self.meta = self._load_meta()
        # Load cached index if available, otherwise build fresh
        if os.path.exists(INDEX_FILE):
            self.index = faiss.read_index(INDEX_FILE)
        else:
            self._build_index() 
        # Pre-compute normalized centroids for tagging
        self._build_centroids() 
    def _load_meta(self):
        with open(META_FILE, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            print(f"Loaded {len(rows)} products from metadata")
            return rows

    def _build_index(self):
        vectors = []
        for r in self.meta:
            v = np.fromstring(r["embedding"], sep=" ").astype("float32")
            v = v / np.linalg.norm(v)   # normalize stored vectors
            vectors.append(v)

        vectors = np.vstack(vectors)
        self.index = faiss.IndexFlatIP(DIM)
        self.index.add(vectors)
        faiss.write_index(self.index, INDEX_FILE)

        print(f"[VectorStore] Built FAISS index with {self.index.ntotal} vectors")

    def _build_centroids(self):
        clusters = defaultdict(list)
        for r in self.meta:
            v = np.fromstring(r["embedding"], sep=" ").astype("float32")
            v = v / np.linalg.norm(v)   # normalize before clustering
            clusters[r["style"]].append(v)
        self.centroids = {}
        for style, vecs in clusters.items():
            c = np.mean(vecs, axis=0)
            c = c / np.linalg.norm(c)   # normalize centroid
            self.centroids[style] = c
        np.save(CENTROID_FILE, self.centroids)
        print(f"Built {len(self.centroids)} normalized style centroids")

    def search(self, q, k=40):
        # Normalize query to match index math
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        D, I = self.index.search(q, k)
        return [(self.meta[i], float(score)) for i, score in zip(I[0], D[0])]
   
    def classify(self, q):
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        best_style, best_score = None, -1
        for style, centroid in self.centroids.items():
            score = float(np.dot(q[0], centroid))
            if score > best_score:
                best_style, best_score = style, score

        return best_style
