from __future__ import annotations
import json
from pathlib import Path

IN_FILES = [
    Path("data/recommendation_training.jsonl"),
    Path("data/recommendation_training_1000.jsonl"),
]
OUT_FILE = Path("data/recommendation_training_all.jsonl")


def read_jsonl(p: Path):
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def main():
    all_rows = []
    for p in IN_FILES:
        all_rows.extend(read_jsonl(p))
    # dedupe by normalized query
    seen = set()
    deduped = []
    for r in all_rows:
        q = (r.get("query") or "").strip().lower()
        if not q or q in seen:
            continue
        seen.add(q)
        # normalize label list order and uniqueness
        areas = sorted(set(r.get("areas") or []))
        deduped.append({"query": r.get("query"), "areas": areas})
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Merged {len(all_rows)} -> {len(deduped)} unique queries into {OUT_FILE}")


if __name__ == "__main__":
    main()
