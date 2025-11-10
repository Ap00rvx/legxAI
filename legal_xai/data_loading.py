import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

DATA_DIR = Path("data")

def _read_json_array(fp: Path) -> List[Dict[str, Any]]:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_jsonl(fp: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def load_corpora(limit: int | None = None) -> pd.DataFrame:
    """Load all available QA corpora into a unified DataFrame.

    Columns: id, source, text
    """
    sources: List[Dict[str, Any]] = []
    mapping = [
        ("constitution", DATA_DIR / "constitution_qa.json", "answer"),
        ("ipc", DATA_DIR / "ipc_qa.json", "answer"),
        ("crpc", DATA_DIR / "crpc_qa.json", "answer"),
    ]
    for name, path, key in mapping:
        if path.exists():
            try:
                arr = _read_json_array(path)
                for i, obj in enumerate(arr):
                    if key in obj:
                        sources.append({
                            "id": f"{name}-{i}",
                            "source": name,
                            "text": obj[key]
                        })
            except Exception:
                # Silently skip malformed items for robustness
                pass

    # Support both train.jsonl (JSON Lines) and train.json (array), and multiple text keys
    def _append_train_rows(objs: List[Dict[str, Any]]):
        for i, obj in enumerate(objs):
            text = None
            for key in ("Response", "answer", "text"):
                if key in obj and isinstance(obj[key], str):
                    text = obj[key]
                    break
            if text:
                sources.append({
                    "id": f"train-{len([s for s in sources if s['source']=='train'])}",
                    "source": "train",
                    "text": text,
                })

    train_jsonl = DATA_DIR / "train.jsonl"
    if train_jsonl.exists():
        try:
            arr = _read_jsonl(train_jsonl)
            _append_train_rows(arr)
        except Exception:
            # Fallback: if file is actually a JSON array with .jsonl extension
            try:
                arr = _read_json_array(train_jsonl)
                _append_train_rows(arr)
            except Exception:
                pass

    train_json = DATA_DIR / "train.json"
    if train_json.exists():
        try:
            arr = _read_json_array(train_json)
            _append_train_rows(arr)
        except Exception:
            pass

    df = pd.DataFrame(sources)
    if limit:
        df = df.head(limit)
    return df

LEGAL_PRACTICE_AREAS = {
    # Expanded keywords and common variants to improve hit rate
    "criminal": [
        "ipc", "crpc", "arrest", "arrested", "warrant", "police", "custody", "remand",
        "penal", "offence", "offense", "trial", "imprisonment", "crime", "punishment",
        "bail", "fir", "charge", "chargesheet", "investigation", "assault", "theft",
        "498a", "420"
    ],
    "constitutional": [
        "constitution", "fundamental right", "fundamental rights", "article 14", "article 19",
        "article ", "equality", "parliament", "writ", "habeas", "mandamus", "supreme court", "high court"
    ],
    "family": [
        "divorce", "custody", "maintenance", "marriage", "alimony", "domestic violence",
        "dowry", "498a", "guardianship"
    ],
    "property": [
        "property", "land", "sale deed", "possession", "title", "encroachment", "mutation",
        "eviction", "lease", "tenancy", "rent"
    ],
    "corporate": [
        "company", "shareholder", "director", "merger", "corporate", "contract", "mou",
        "compliance", "ipo", "esop"
    ],
    "civil": ["civil", "plaint", "injunction", "suit", "damages", "decree"],
}

_area_pattern_cache: Dict[str, re.Pattern] = {}
for area, kws in LEGAL_PRACTICE_AREAS.items():
    toks = []
    for k in kws:
        k_esc = re.escape(k)
        # Add word boundaries for very short tokens to avoid false positives (e.g., 'fir' in 'first')
        if len(k) <= 3 or k.lower() in {"fir", "ipc", "crpc", "act", "sc", "hc"}:
            toks.append(r"\b" + k_esc + r"\b")
        else:
            toks.append(k_esc)
    pattern = r"(" + r"|".join(toks) + r")"
    _area_pattern_cache[area] = re.compile(pattern, re.IGNORECASE)


def infer_practice_areas(text: str) -> List[str]:
    """Infer candidate practice areas from a query string (rule-based).

    Returns ordered unique areas based on regex hits; falls back to a guess if none matched.
    """
    hits: List[str] = []
    for area, pat in _area_pattern_cache.items():
        if pat.search(text or ""):
            hits.append(area)
    # Deduplicate while preserving order
    seen = set()
    hits = [h for h in hits if not (h in seen or seen.add(h))]
    # fallback classification if none matched
    if not hits:
        # If query contains generic legal cues, default to constitutional else civil
        t = (text or "").lower()
        generic_legal = ["law", "legal", "court", "judge", "act", "section", "article", "petition", "writ"]
        if any(g in t for g in generic_legal):
            hits.append("constitutional")
        else:
            hits.append("civil")
    return hits

__all__ = ["load_corpora", "infer_practice_areas", "LEGAL_PRACTICE_AREAS"]
