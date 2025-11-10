from __future__ import annotations
import json
import random
from pathlib import Path

OUT_PATH = Path("data/recommendation_training_1000.jsonl")
random.seed(42)

AREAS = {
    "criminal": {
        "syn": [
            "arrest", "warrant", "FIR", "bail", "custody", "remand", "police",
            "IPC 420", "IPC 406", "498A", "assault", "theft", "cheating",
            "chargesheet", "ED summons", "money laundering",
        ],
        "templates": [
            "Police {verb} my {relation} without {doc} at night",
            "Need {adj} bail for {section}",
            "File {doc} refusal by police, next steps",
            "Defense for {section} allegation in {context}",
        ],
        "fill": {
            "verb": ["arrested", "detained"],
            "relation": ["brother", "friend", "husband", "son"],
            "doc": ["warrant", "proper grounds"],
            "adj": ["anticipatory", "regular"],
            "section": ["498A", "IPC 420", "IPC 406"],
            "context": ["business dispute", "family matter", "neighbour fight"],
        },
    },
    "constitutional": {
        "syn": [
            "writ", "habeas corpus", "mandamus", "Article 14", "Article 19",
            "fundamental rights", "Supreme Court", "High Court", "RTI",
        ],
        "templates": [
            "Seek {writ} against government {issue}",
            "{article} violation by local authority during {context}",
            "High Court {writ} for {service} matter",
        ],
        "fill": {
            "writ": ["writ of mandamus", "habeas corpus", "certiorari"],
            "issue": ["inaction", "illegal order", "delay"],
            "article": ["Article 14", "Article 19", "Article 21"],
            "context": ["protest", "seizure", "curfew"],
            "service": ["promotion denial", "transfer", "suspension"],
        },
    },
    "family": {
        "syn": [
            "divorce", "custody", "maintenance", "alimony", "domestic violence",
            "dowry", "guardianship", "498A",
        ],
        "templates": [
            "{topic} after separation, need legal process",
            "Protection under {act} and evidence collection",
            "Child {issue} dispute with {relation}",
        ],
        "fill": {
            "topic": ["Mutual consent divorce", "Maintenance claim", "Alimony"],
            "act": ["domestic violence act", "dowry harassment"],
            "issue": ["custody", "support"],
            "relation": ["spouse", "in-laws"],
        },
    },
    "property": {
        "syn": [
            "sale deed", "possession", "title", "encroachment", "mutation",
            "eviction", "lease", "tenancy", "rent",
        ],
        "templates": [
            "Dispute over {doc} and {proc} entry",
            "{action} notice for {reason} in rented house",
            "Injunction to stop {encroach} on boundary",
        ],
        "fill": {
            "doc": ["land sale deed", "property documents"],
            "proc": ["mutation", "revenue"],
            "action": ["Eviction", "Vacate"],
            "reason": ["non payment of rent", "unauthorized construction"],
            "encroach": ["encroachment", "illegal construction"],
        },
    },
    "corporate": {
        "syn": [
            "shareholder", "director", "merger", "compliance", "term sheet",
            "ESOP", "IPO", "GST", "arbitration",
        ],
        "templates": [
            "{event} and {topic} review for board",
            "Draft {doc} for startup {context}",
            "{law} penalty challenge and representation",
        ],
        "fill": {
            "event": ["Merger", "Acquisition"],
            "topic": ["compliance", "shareholder rights"],
            "doc": ["founders agreement", "shareholders agreement"],
            "context": ["investment", "ESOP plan"],
            "law": ["GST", "Companies Act"],
        },
    },
    "civil": {
        "syn": [
            "injunction", "plaint", "damages", "decree", "specific performance",
            "contract breach", "money recovery",
        ],
        "templates": [
            "{remedy} to stop {issue} by neighbour",
            "{claim} for {context} without written agreement",
            "{suit} for {subject} in civil court",
        ],
        "fill": {
            "remedy": ["Temporary injunction", "Stay order"],
            "issue": ["construction", "trespass"],
            "claim": ["Recovery of money", "Damages"],
            "context": ["loan", "services"],
            "suit": ["Suit", "Plaint"],
            "subject": ["specific performance", "defamation"],
        },
    },
}


def sample_single(area: str) -> str:
    cfg = AREAS[area]
    tpl = random.choice(cfg["templates"])
    fill = {k: random.choice(v) for k, v in cfg["fill"].items()}
    q = tpl.format(**fill)
    # occasionally append a synonym for diversity
    if random.random() < 0.4:
        q += f" ({random.choice(cfg['syn'])})"
    return q


def sample_multi(areas: list[str]) -> str:
    # join two area samples into one mixed query with a connector
    a1, a2 = areas
    q1 = sample_single(a1)
    q2 = sample_single(a2)
    connector = random.choice([
        "Also need help with", "Additionally", "Plus", "Meanwhile",
    ])
    return f"{q1}. {connector} {q2.lower()}"


def generate(n: int = 1000) -> list[dict]:
    out = []
    area_list = list(AREAS.keys())
    for i in range(n):
        mode = random.random()
        if mode < 0.7:
            # single-label 70%
            a = random.choice(area_list)
            q = sample_single(a)
            out.append({"query": q, "areas": [a]})
        else:
            # multi-label 30%
            a1, a2 = random.sample(area_list, 2)
            q = sample_multi([a1, a2])
            out.append({"query": q, "areas": [a1, a2]})
    return out


def main():
    data = generate(1000)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # print brief stats
    from collections import Counter
    c = Counter()
    for r in data:
        for a in r["areas"]:
            c[a] += 1
    print("Wrote:", OUT_PATH, "Total:", len(data))
    print("Label counts:", dict(c))


if __name__ == "__main__":
    main()
