from __future__ import annotations
from typing import Dict, Tuple, Optional
from threading import Lock

import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# Centralized, singleton-style model registry to prevent duplicate loads

"""Model registry: now supports overriding QA model with a legal-domain BERT via env vars.

Set LEGAL_QA_MODEL or QA_MODEL_NAME environment variable to switch the QA backbone, e.g.:
    LEGAL_QA_MODEL=law-ai/InLegalBERT-SQuAD      # if already QA fine-tuned
    LEGAL_QA_MODEL=nlpaueb/legal-bert-base-uncased (requires fine-tuning for QA before use)
Falls back to default SQuAD2 fine-tuned model.
"""

QA_MODEL_NAME = os.getenv("LEGAL_QA_MODEL", os.getenv("QA_MODEL_NAME", "deepset/bert-base-cased-squad2"))

_qa_lock = Lock()
_qa_tokenizer = None
_qa_model = None

_embed_lock = Lock()
_embed_models: Dict[str, SentenceTransformer] = {}

_rec_lock = Lock()
_rec_tokenizer: Optional[AutoTokenizer] = None
_rec_model: Optional[AutoModelForSequenceClassification] = None
_rec_dir: Optional[str] = None

_gen_lock = Lock()
_gen_tokenizer: Optional[AutoTokenizer] = None
_gen_model: Optional[AutoModelForSeq2SeqLM] = None
_gen_name: Optional[str] = None


def get_qa_components(model_name: str = QA_MODEL_NAME) -> Tuple[AutoTokenizer, AutoModelForQuestionAnswering]:
    """Get shared QA tokenizer and model (singleton per process)."""
    global _qa_tokenizer, _qa_model
    if _qa_tokenizer is not None and _qa_model is not None and model_name == QA_MODEL_NAME:
        return _qa_tokenizer, _qa_model
    with _qa_lock:
        if _qa_tokenizer is None or _qa_model is None or model_name != QA_MODEL_NAME:
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


def get_recommend_classifier(model_dir: str = "models/recommend_classifier") -> Optional[Tuple[AutoTokenizer, AutoModelForSequenceClassification]]:
    """Load and cache the fine-tuned recommendation classifier if available.

    Returns (tokenizer, model) or None if the directory doesn't exist.
    """
    global _rec_tokenizer, _rec_model, _rec_dir
    if _rec_tokenizer is not None and _rec_model is not None and _rec_dir == model_dir:
        return _rec_tokenizer, _rec_model
    if not os.path.isdir(model_dir):
        return None
    with _rec_lock:
        if _rec_tokenizer is None or _rec_model is None or _rec_dir != model_dir:
            tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
            mdl = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=False)
            mdl.eval()
            _rec_tokenizer, _rec_model, _rec_dir = tok, mdl, model_dir
    return _rec_tokenizer, _rec_model


def get_generation_model(model_name: str | None = None) -> Optional[Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]]:
    """Load / cache a seq2seq generation model (e.g., Flan-T5) for synthesis.

    Controlled by LEGAL_GEN_MODEL env var; returns None if unavailable.
    """
    import os
    name = model_name or os.getenv("LEGAL_GEN_MODEL") or "google/flan-t5-base"
    global _gen_tokenizer, _gen_model, _gen_name
    if _gen_tokenizer is not None and _gen_model is not None and _gen_name == name:
        return _gen_tokenizer, _gen_model
    with _gen_lock:
        if _gen_tokenizer is None or _gen_model is None or _gen_name != name:
            def _load_target(target_name: str):
                tok_local = AutoTokenizer.from_pretrained(target_name, trust_remote_code=False)
                # Quantization / device options via env vars
                load_kwargs = {}
                device_map = os.getenv("GEN_DEVICE_MAP", "")
                if device_map:
                    load_kwargs["device_map"] = device_map
                use_4bit = os.getenv("GEN_4BIT", os.getenv("GEN_LOAD_IN_4BIT", "0")) in ("1", "true", "True")
                use_8bit = os.getenv("GEN_8BIT", os.getenv("GEN_LOAD_IN_8BIT", "0")) in ("1", "true", "True")
                torch_dtype_env = os.getenv("GEN_TORCH_DTYPE", "")
                if torch_dtype_env.lower() == "fp16":
                    import torch as _torch
                    load_kwargs["torch_dtype"] = _torch.float16
                if use_4bit or use_8bit:
                    try:
                        from transformers import BitsAndBytesConfig  # type: ignore
                        if use_4bit:
                            import torch as _torch
                            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=_torch.float16,
                            )
                        elif use_8bit:
                            load_kwargs["load_in_8bit"] = True
                    except Exception:
                        pass
                mdl_local = AutoModelForSeq2SeqLM.from_pretrained(target_name, trust_remote_code=False, **load_kwargs)
                if os.getenv("GEN_CPU_INT8", "0") in ("1", "true", "True"):
                    try:
                        import torch as _torch
                        mdl_local = mdl_local.cpu()
                        mdl_local = _torch.quantization.quantize_dynamic(mdl_local, { _torch.nn.Linear }, dtype=_torch.qint8)
                    except Exception:
                        pass
                mdl_local.eval()
                return tok_local, mdl_local
            try:
                tok, mdl = _load_target(name)
            except Exception:
                # Fallback to flan-t5-base unquantized
                try:
                    tok, mdl = _load_target("google/flan-t5-base")
                except Exception:
                    return None
            _gen_tokenizer, _gen_model, _gen_name = tok, mdl, name
    return _gen_tokenizer, _gen_model
