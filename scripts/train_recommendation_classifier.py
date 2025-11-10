from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


LABEL_ORDER = ["criminal", "constitutional", "family", "property", "corporate", "civil"]


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def to_multi_hot(areas: List[str]) -> List[int]:
    vec = [0] * len(LABEL_ORDER)
    for a in areas:
        if a in LABEL_ORDER:
            vec[LABEL_ORDER.index(a)] = 1
    return vec


class JsonlDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer: AutoTokenizer, max_length: int = 256):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        text = r.get("query") or ""
        labels = to_multi_hot(r.get("areas") or [])
        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(labels, dtype=torch.float)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/recommendation_training_all.jsonl")
    ap.add_argument("--model", type=str, default="distilbert-base-uncased")
    ap.add_argument("--out", type=str, default="models/recommend_classifier")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--check", action="store_true", help="Only load data and print stats, no training")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        # fall back to smaller file
        alt = Path("data/recommendation_training.jsonl")
        if alt.exists():
            data_path = alt
        else:
            raise FileNotFoundError(f"No dataset found at {args.data} or {alt}")

    rows = load_jsonl(data_path)
    # sanitize
    clean = []
    for r in rows:
        q = (r.get("query") or "").strip()
        areas = sorted(set([a for a in (r.get("areas") or []) if a in LABEL_ORDER]))
        if not q or not areas:
            continue
        clean.append({"query": q, "areas": areas})

    print(f"Loaded {len(clean)} samples from {data_path}")
    label_counts = {lab: 0 for lab in LABEL_ORDER}
    for r in clean:
        for a in r["areas"]:
            label_counts[a] += 1
    print("Label counts:", label_counts)

    if args.check:
        return

    train_rows, val_rows = train_test_split(clean, test_size=0.15, random_state=42)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL_ORDER),
        problem_type="multi_label_classification",
    )

    train_ds = JsonlDataset(train_rows, tok)
    val_ds = JsonlDataset(val_rows, tok)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    # save label order for inference
    with (out_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump({"labels": LABEL_ORDER}, f, ensure_ascii=False, indent=2)

    print("Training complete. Model saved to", out_dir)


if __name__ == "__main__":
    main()
