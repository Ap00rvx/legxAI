from __future__ import annotations
from typing import Dict, Tuple
from threading import Lock

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer

# Centralized, singleton-style model registry to prevent duplicate loads

QA_MODEL_NAME = "deepset/bert-base-cased-squad2"

_qa_lock = Lock()
_qa_tokenizer = None
_qa_model = None

_embed_lock = Lock()
_embed_models: Dict[str, SentenceTransformer] = {}


def get_qa_components(model_name: str = QA_MODEL_NAME) -> Tuple[AutoTokenizer, AutoModelForQuestionAnswering]:
    """Get shared QA tokenizer and model (singleton per process)."""
    global _qa_tokenizer, _qa_model
    if _qa_tokenizer is not None and _qa_model is not None:
        return _qa_tokenizer, _qa_model
    with _qa_lock:
        if _qa_tokenizer is None or _qa_model is None:
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
            mdl = AutoModelForQuestionAnswering.from_pretrained(model_name, trust_remote_code=False)
            mdl.eval()
            _qa_tokenizer, _qa_model = tok, mdl
    return _qa_tokenizer, _qa_model


def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Get (and cache) a SentenceTransformer model by name."""
    if model_name in _embed_models:
        return _embed_models[model_name]
    with _embed_lock:
        if model_name not in _embed_models:
            _embed_models[model_name] = SentenceTransformer(model_name)
    return _embed_models[model_name]
