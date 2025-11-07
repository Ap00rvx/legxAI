from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from .model_registry import get_embedding_model


DEFAULT_EMBED_MODEL = "sentence-transformers/msmarco-bert-base-dot-v5"  # BERT-based dense retriever


@dataclass
class TextIndex:
    ids: List[str]
    texts: List[str]
    embeddings: np.ndarray  # shape (N, D)
    model_name: str

    def search(self, query: str, top_k: int = 5, model: Optional[SentenceTransformer] = None):
        if model is None:
            model = get_embedding_model(self.model_name)
        q = model.encode([query], normalize_embeddings=True)
        doc = self.embeddings
        sims = (doc @ q.T).squeeze(1)
        idx = np.argsort(-sims)[:top_k]
        results = []
        for i in idx:
            results.append({
                "id": self.ids[i],
                "text": self.texts[i],
                "score": float(sims[i]),
            })
        return results


def build_index(ids: List[str], texts: List[str], model_name: str = DEFAULT_EMBED_MODEL) -> TextIndex:
    model = get_embedding_model(model_name)
    emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return TextIndex(ids=ids, texts=texts, embeddings=emb, model_name=model_name)
