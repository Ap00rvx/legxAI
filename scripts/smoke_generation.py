"""Smoke test for synthesized legal answers.

Runs a few queries through retrieval + generation module directly (not via API).
Ensure LEGAL_GEN_MODEL is set (e.g., google/flan-t5-base). Falls back to flan-t5-small.
"""
from pathlib import Path
import sys

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from legal_xai.data_loading import load_corpora
from legal_xai.retrieval import build_corpus_index, retrieve_top_k
from legal_xai.generation import synthesize_answer

QUERIES = [
    "What is Article 14 and how is equality before law applied?",
    "Explain anticipatory bail procedure under Indian law",
    "Difference between eviction and injunction in property disputes",
]

def main():
    df = load_corpora(limit=150)
    index = build_corpus_index(df)
    for q in QUERIES:
        passages = retrieve_top_k(index, q, k=3)
        synth = synthesize_answer(q, passages, style="concise")
        print("Query:", q)
        print("Synthesized:", synth.get("text"))
        print("Citations:", synth.get("citations"))
        print("Low confidence:", synth.get("low_confidence"))
        print("---")

if __name__ == "__main__":
    main()
