from __future__ import annotations
from typing import List, Dict
import pandas as pd
from .embedding import build_index, TextIndex, DEFAULT_EMBED_MODEL


def build_corpus_index(df: pd.DataFrame, model_name: str = DEFAULT_EMBED_MODEL) -> TextIndex:
    # Clean and filter
    pdf = df.dropna(subset=["text"]).copy()
    pdf["text"] = pdf["text"].astype(str).str.strip()
    pdf = pdf[pdf["text"].str.len() > 20]
    return build_index(ids=pdf["id"].tolist(), texts=pdf["text"].tolist(), model_name=model_name)


def retrieve_top_k(index: TextIndex, query: str, k: int = 5) -> List[Dict]:
    return index.search(query, top_k=k)
