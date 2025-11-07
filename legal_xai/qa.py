from __future__ import annotations
from typing import Dict, Any, List
from .model_registry import get_qa_components
import torch


def extract_answer(question: str, context: str) -> Dict[str, Any]:
    tokenizer, model = get_qa_components()
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    start_scores = out.start_logits
    end_scores = out.end_logits
    start_idx = int(torch.argmax(start_scores))
    end_idx = int(torch.argmax(end_scores))
    all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    if end_idx < start_idx:
        end_idx = start_idx
    answer = tokenizer.convert_tokens_to_string(all_tokens[start_idx:end_idx+1])
    score = (float(torch.max(start_scores)) + float(torch.max(end_scores))) / 2.0
    return {"answer": answer, "score": score, "start": start_idx, "end": end_idx}


def batch_answer(question: str, passages: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    answers = []
    for p in passages[:top_n]:
        ans = extract_answer(question, p["text"])
        answers.append({"passage_id": p["id"], "passage_score": p.get("score"), **ans, "passage": p["text"]})
    # sort by QA confidence
    answers.sort(key=lambda x: x["score"], reverse=True)
    return answers
