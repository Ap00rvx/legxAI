from __future__ import annotations
import json
import random
import re
import time
import sys
from pathlib import Path
from typing import List, Dict

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from legal_xai.data_loading import load_corpora
from legal_xai.retrieval import build_corpus_index, retrieve_top_k
from legal_xai.qa import batch_answer


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s


def f1_em(pred: str, gold: str) -> (float, float):
    p = normalize_text(pred)
    g = normalize_text(gold)
    # EM
    em = 1.0 if p == g else 0.0
    # token F1
    p_tokens = p.split()
    g_tokens = g.split()
    if not p_tokens and not g_tokens:
        return 1.0, 1.0
    if not p_tokens or not g_tokens:
        return 0.0, em
    common = {}
    for t in p_tokens:
        common[t] = min(p_tokens.count(t), g_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, em
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, em


def evaluate(sample_size: int = 200, k: int = 3) -> Dict[str, float]:
    df = load_corpora(limit=10000)
    index = build_corpus_index(df)
    # Build a simple synthetic evaluation set from our corpora-style QAs if present
    eval_set: List[Dict[str, str]] = []
    for col in ("constitution_qa", "ipc_qa", "crpc_qa"):
        items = df[df["source"].str.contains(col, na=False)]
        for _, row in items.iterrows():
            if isinstance(row.get("qa"), dict):
                q = row["qa"].get("question")
                a = row["qa"].get("answer")
                if q and a:
                    eval_set.append({"question": q, "answer": a})
    if not eval_set:
        # fallback small set from files
        base = Path("data")
        for name in ["constitution_qa.json", "ipc_qa.json", "crpc_qa.json"]:
            p = base / name
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                for r in data:
                    eval_set.append({"question": r["question"], "answer": r["answer"]})
    random.seed(42)
    random.shuffle(eval_set)
    eval_set = eval_set[:sample_size]

    start = time.time()
    total_f1 = 0.0
    total_em = 0.0
    count = 0
    for item in eval_set:
        q = item["question"]
        gold = item["answer"]
        passages = retrieve_top_k(index, q, k=k)
        answers = batch_answer(q, passages, top_n=k)
        pred = answers[0]["answer"] if answers else ""
        f1, em = f1_em(pred, gold)
        total_f1 += f1
        total_em += em
        count += 1
    dur = time.time() - start
    return {
        "samples": count,
        "f1": round(total_f1 / max(1, count), 4),
        "em": round(total_em / max(1, count), 4),
        "seconds": round(dur, 2),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of samples")
    ap.add_argument("--k", type=int, default=3, help="top-k passages and answers")
    args = ap.parse_args()
    metrics = evaluate(sample_size=args.n, k=args.k)
    print(metrics)
