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
    "criminal": ["ipc", "crpc", "arrest", "penal", "offence", "trial", "imprisonment", "crime", "punishment"],
    "constitutional": ["constitution", "fundamental right", "article 14", "article 19", "equality", "parliament"],
    "family": ["divorce", "custody", "maintenance", "marriage"],
    "property": ["property", "land", "sale deed", "possession", "title"],
    "corporate": ["company", "shareholder", "director", "merger", "corporate"],
    "civil": ["civil", "plaint", "injunction", "suit", "damages"],
}

_area_pattern_cache: Dict[str, re.Pattern] = {}
for area, kws in LEGAL_PRACTICE_AREAS.items():
    pattern = r"(" + r"|".join(re.escape(k) for k in kws) + r")"
    _area_pattern_cache[area] = re.compile(pattern, re.IGNORECASE)


def infer_practice_areas(text: str) -> List[str]:
    """Infer candidate practice areas from a query string (rule-based)."""
    hits = []
    for area, pat in _area_pattern_cache.items():
        if pat.search(text):
            hits.append(area)
    # fallback classification if none matched
    if not hits:
        # heuristic: length-based guess
        if len(text) < 120:
            hits.append("civil")
        else:
            hits.append("constitutional")
    return hits

__all__ = ["load_corpora", "infer_practice_areas", "LEGAL_PRACTICE_AREAS"]
