# app/spacy_embed.py
from __future__ import annotations
from typing import List, Tuple, Dict
import os
import numpy as np
import spacy


class SpaCyEmbedder:

    def __init__(self, model_name: str | None = None):
        name = model_name or os.getenv("SPACY_MODEL", "en_core_web_lg")
        # This will raise if the model isn't installed; install instructions below.
        self.nlp = spacy.load(name)

    # ---- vectors ----
    def vector(self, word: str) -> np.ndarray:
        # The notes use nlp(word).vector for a single token. Works in spaCy v3.
        doc = self.nlp(word)
        return doc.vector  # 1D numpy array

    # ---- neighbors ----
    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float((a @ b) / (na * nb))

    def nearest_neighbors(self, word: str, k: int = 10) -> List[Dict]:
        """
        Top-k neighbors by cosine similarity using spaCy's vocab vectors.
        """
        query = self.vector(word)
        if query.size == 0:
            return []

        # Use spaCy's shared vector table
        vectors = self.nlp.vocab.vectors
        if vectors is None or vectors.data.size == 0:
            return []

        mat = vectors.data  # shape: (n_vectors, dim)
        keys = vectors.keys()
        strings = self.nlp.vocab.strings

        # normalize for cosine similarity
        qn = query / (np.linalg.norm(query) + 1e-12)
        mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        sims = mn @ qn  # (n_vectors,)

        # map back to string forms and take top-k (skip exact same token)
        order = np.argsort(-sims)
        out: List[Dict] = []
        for i in order:
            token = strings[keys[i]]
            if token.lower() == word.lower():
                continue
            out.append({"token": token, "similarity": float(sims[i])})
            if len(out) == k:
                break
        return out