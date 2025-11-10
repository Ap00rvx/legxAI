from __future__ import annotations
import os
import re
from typing import List, Dict, Any, Tuple

import torch

from .model_registry import get_generation_model

MAX_PASSAGE_CHARS = int(os.getenv("GEN_PASSAGE_TRIM", "450"))
DEFAULT_GEN_MODEL = os.getenv("LEGAL_GEN_MODEL", "google/flan-t5-small")


def _trim(text: str, limit: int = MAX_PASSAGE_CHARS) -> str:
    if len(text) <= limit:
        return text.strip()
    return text[:limit].rsplit(" ", 1)[0].strip() + "…"


def prepare_prompt(question: str, passages: List[Dict[str, Any]], style: str = "concise") -> Tuple[str, List[Tuple[str, str]]]:
    numbered: List[Tuple[str, str]] = []
    for i, p in enumerate(passages, start=1):
        pid = f"P{i}"
        txt = _trim(p.get("text", ""))
        numbered.append((pid, txt))
    # Determine style guidance dynamically
    env_style = os.getenv("GEN_STYLE") or style
    norm_style = (env_style or "concise").lower()
    if norm_style == "detailed":
        guidance = "Provide a detailed answer (6-12 sentences: definition, scope, key principles, procedural aspects, exceptions, concluding clarification)."
    elif norm_style == "bullet":
        guidance = "Answer using bullet points: definition; key elements; procedure; exceptions; practical considerations. 6-10 bullets."  
    else:
        guidance = "Provide a concise answer (3-6 sentences)."
    inst = [
        "You are an Indian legal assistant.",
        guidance,
        "Cite sources using their bracket ids like [P1] [P2].",
        "Start with a direct answer sentence before elaboration.",
        "Only use provided passages; if insufficient, say you need more details.",
        "Avoid hallucinating Articles or Sections not present in passages.",
    ]
    prompt = "\n".join(inst) + "\n\nQuestion: " + question.strip() + "\n\nPassages:\n"
    for pid, txt in numbered:
        prompt += f"[{pid}] {txt}\n"
    prompt += "\nAnswer:"
    return prompt, numbered


def _extract_citations(text: str) -> List[str]:
    found = re.findall(r"\[(P\d+)\]", text)
    return sorted(set(found), key=lambda x: int(x[1:]))


def synthesize_answer(question: str, passages: List[Dict[str, Any]], style: str = "detailed", model_name: str | None = None, max_new_tokens: int = 512) -> Dict[str, Any]:
    comp = get_generation_model(model_name or DEFAULT_GEN_MODEL)
    if comp is None:
        return {"text": "Synthesis model unavailable.", "citations": [], "low_confidence": True}
    tok, model = comp
    # For longer answers, adjust style instruction
    style_instruction = style or "detailed"
    prompt, numbered = prepare_prompt(question, passages, style=style_instruction)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=float(os.getenv("GEN_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("GEN_TOP_P", "0.9")),
            num_beams=int(os.getenv("GEN_BEAMS", "1")),
            do_sample=os.getenv("GEN_DO_SAMPLE", "1") in ("1","true","True"),
            length_penalty=float(os.getenv("GEN_LENGTH_PENALTY", "1.0")),
        )
    text = tok.decode(gen_ids[0], skip_special_tokens=True).strip()
    # Remove potential duplicated prompt echoes
    if "Answer:" in text:
        text = text.split("Answer:", 1)[-1].strip()
    citations = _extract_citations(text)
    # Build citation mapping
    id_to_passage = {pid: txt for pid, txt in numbered}
    cited_texts = [{"id": cid, "text": id_to_passage.get(cid, "")} for cid in citations]
    # Confidence heuristic: require at least 1 citation and length threshold
    alpha_len = len(re.sub(r"[^A-Za-z]", "", text))
    low_conf = len(citations) == 0 or alpha_len < 80
    return {
        "text": text,
        "citations": citations,
        "citation_passages": cited_texts,
        "low_confidence": low_conf,
    }


__all__ = ["synthesize_answer"]
