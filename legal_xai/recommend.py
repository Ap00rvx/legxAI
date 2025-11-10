from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np

from .data_loading import LEGAL_PRACTICE_AREAS
from .embedding import DEFAULT_EMBED_MODEL
from .model_registry import get_embedding_model, get_recommend_classifier


LAWYER_LABELS = {
    "criminal": "Criminal Lawyer",
    "constitutional": "Constitutional / Public Law Lawyer",
    "family": "Family Lawyer",
    "property": "Property / Real Estate Lawyer",
    "corporate": "Corporate / Business Lawyer",
    "civil": "Civil Litigation Lawyer",
}


# Concise area descriptions for embedding-based matching (expanded with India-specific cues)
AREA_DESCRIPTIONS = {
    "criminal": (
        "Indian criminal matters: IPC sections (302 murder, 420 cheating, 498A cruelty), CrPC procedure (arrest, remand, charge sheet), FIR filing, bail (regular/anticipatory), police investigation, offences and punishment."),
    "constitutional": (
        "Constitutional & public law: Fundamental Rights (Articles 14 equality, 19 freedoms, 21 life & liberty), writs (habeas corpus, mandamus), Supreme Court / High Court jurisdiction, due process and state action."),
    "family": (
        "Family & personal law: marriage, divorce, custody, maintenance, alimony, domestic violence, dowry issues (498A context), guardianship under Hindu & secular statutes."),
    "property": (
        "Property / real estate: land title, sale deed, encroachment, possession disputes, tenancy & lease, eviction under Rent Control, mutation & registration, adverse possession claims."),
    "corporate": (
        "Corporate & business: company law (shareholders, directors), MCA compliance, contracts & MOUs, mergers & acquisitions, governance, regulatory filings."),
    "civil": (
        "Civil litigation: plaint drafting, injunctions (temporary/permanent), suits for damages, decrees & execution, general non-criminal private disputes."),
}


_area_ids: List[str] = list(AREA_DESCRIPTIONS.keys())
_area_texts: List[str] = [AREA_DESCRIPTIONS[a] for a in _area_ids]
_area_embs: np.ndarray | None = None
_area_model_name: str | None = None
_clf_labels: List[str] | None = None


def _ensure_area_embeddings(model_name: str = DEFAULT_EMBED_MODEL) -> Tuple[np.ndarray, List[str]]:
    global _area_embs, _area_model_name
    if (_area_embs is None) or (_area_model_name != model_name):
        model = get_embedding_model(model_name)
        embs = model.encode(_area_texts, normalize_embeddings=True, convert_to_numpy=True)
        _area_embs = embs.astype(np.float32)
        _area_model_name = model_name
    return _area_embs, _area_ids


def _rank_areas_by_similarity(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    embs, area_ids = _ensure_area_embeddings()
    model = get_embedding_model(_area_model_name or DEFAULT_EMBED_MODEL)
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)[0]
    sims = (embs @ q)
    order = np.argsort(-sims)[:top_k]
    return [(area_ids[i], float(sims[i])) for i in order]


def _softmax(vec: np.ndarray) -> np.ndarray:
    if vec.size == 0:
        return vec
    v = vec - np.max(vec)
    e = np.exp(v)
    return e / (e.sum() + 1e-9)


def _clarifying_questions(a: str, b: str) -> List[str]:
    pairs = {
        ("criminal", "family"): ["Is the core issue police procedure or family relations?", "Does it involve domestic violence charges or custody/maintenance?"],
        ("criminal", "civil"): ["Are you seeking punishment or civil damages?", "Is it an IPC offence or a private dispute?"],
        ("property", "civil"): ["Is this primarily a property title/possession matter or a broader civil claim?", "Are you contesting ownership or seeking general relief?"],
        ("constitutional", "civil"): ["Are fundamental rights involved or is it a private civil dispute?", "Do you plan a writ petition or a regular suit?"],
        ("corporate", "civil"): ["Is this about company governance/compliance or a general civil issue?", "Are corporate filings or contracts central?"],
        ("family", "criminal"): ["Is it about domestic violence allegations or family law remedies?", "Do you need protective action or matrimonial relief?"],
    }
    key = (a, b)
    if key not in pairs:
        key = (b, a)
    return pairs.get(key, ["Could you clarify the core nature of the dispute?", "Is the matter procedural or rights-based?"])


def _predict_with_classifier(query: str) -> Dict[str, float]:
    comp = get_recommend_classifier()
    if comp is None:
        return {}
    tok, model = comp
    enc = tok(query, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():  # type: ignore
        out = model(**{k: v for k, v in enc.items()})
        logits = out.logits.squeeze(0).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    # Load label order
    global _clf_labels
    if _clf_labels is None:
        import json, os
        path = os.path.join("models", "recommend_classifier", "labels.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                _clf_labels = json.load(f).get("labels")
        except Exception:
            _clf_labels = list(AREA_DESCRIPTIONS.keys())
    return {lab: float(prob) for lab, prob in zip(_clf_labels, probs)}


def recommend_lawyers(query: str, top_k: int = 3, use_classifier: bool = True) -> Dict[str, any]:
    """Embedding-based recommendation with confidence & margin.

    Returns:
    {
      recommendations: [ {area,label,keywords,score,confidence} ],
      margin: float,               # top1 - top2 raw similarity
      clarification_needed: bool,  # margin below threshold
      clarifying_questions: [str]
    }
    """
    ranked = _rank_areas_by_similarity(query, top_k=top_k)
    sims = np.array([s for _, s in ranked], dtype=np.float32)

    clf_probs = _predict_with_classifier(query) if use_classifier else {}
    # Fuse embedding similarity with classifier probability if available:
    # simple average after aligning keys.
    fused_scores = []
    for area, sim in ranked:
        if clf_probs:
            prob = clf_probs.get(area, 0.0)
            score = (sim + prob) / 2.0
        else:
            score = sim
        fused_scores.append((area, score, sim, clf_probs.get(area, None)))
    # Re-rank on fused score
    fused_scores.sort(key=lambda x: x[1], reverse=True)
    ranked = [(a, s) for a, s, _, _ in fused_scores]
    sims = np.array([s for _, s in ranked], dtype=np.float32)
    confs = _softmax(sims)
    recs: List[Dict[str, any]] = []
    for (area, score), conf in zip(ranked, confs):
        recs.append({
            "area": area,
            "label": LAWYER_LABELS.get(area, area.title() + " Lawyer"),
            "keywords": ", ".join(LEGAL_PRACTICE_AREAS.get(area, [])),
            "score": round(float(score), 4),
            "confidence": round(float(conf), 4),
            "raw_similarity": round(float(next((orig for a2, orig, _, _ in fused_scores if a2 == area), score)), 4),
            "classifier_prob": round(float(next((prob for a2, _, _, prob in fused_scores if a2 == area and prob is not None), 0.0)), 4),
        })
    margin = float(sims[0] - sims[1]) if sims.size >= 2 else 0.0
    clarification_needed = sims.size >= 2 and margin < 0.03
    clarifying_questions = _clarifying_questions(ranked[0][0], ranked[1][0]) if clarification_needed else []
    return {
        "recommendations": recs,
        "margin": round(margin, 4),
        "clarification_needed": clarification_needed,
        "clarifying_questions": clarifying_questions,
    }


__all__ = ["recommend_lawyers", "LAWYER_LABELS", "AREA_DESCRIPTIONS"]
