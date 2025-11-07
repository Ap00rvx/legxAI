from __future__ import annotations
from typing import List, Dict
from .data_loading import infer_practice_areas, LEGAL_PRACTICE_AREAS

LAWYER_LABELS = {
    "criminal": "Criminal Lawyer",
    "constitutional": "Constitutional / Public Law Lawyer",
    "family": "Family Lawyer",
    "property": "Property / Real Estate Lawyer",
    "corporate": "Corporate / Business Lawyer",
    "civil": "Civil Litigation Lawyer",
}


def recommend_lawyers(query: str) -> List[Dict[str, str]]:
    areas = infer_practice_areas(query)
    recs: List[Dict[str, str]] = []
    for a in areas:
        recs.append({
            "area": a,
            "label": LAWYER_LABELS.get(a, a.title() + " Lawyer"),
            "keywords": ", ".join(LEGAL_PRACTICE_AREAS.get(a, []))
        })
    return recs

__all__ = ["recommend_lawyers", "LAWYER_LABELS"]
