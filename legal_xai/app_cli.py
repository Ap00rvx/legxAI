from __future__ import annotations
import argparse
import json
import os
from .data_loading import load_corpora
from .retrieval import build_corpus_index, retrieve_top_k
from .qa import batch_answer
from .explain import explain_qa
from .recommend import recommend_lawyers

try:
    # Optional pretty printing with colors
    from rich.console import Console
    from rich.text import Text
    _RICH_AVAILABLE = True
except Exception:
    _RICH_AVAILABLE = False

_index_cache = None

def ensure_index():
    global _index_cache
    if _index_cache is None:
        # Limit index size by default for faster first-run; override via env var
        # Set LEGAL_XAI_INDEX_LIMIT to control number of passages indexed
        index_limit = int(os.environ.get("LEGAL_XAI_INDEX_LIMIT", "800"))
        df = load_corpora(limit=index_limit)
        _index_cache = build_corpus_index(df)
    return _index_cache


def cmd_ask(args):
    index = ensure_index()
    passages = retrieve_top_k(index, args.question, k=args.top_k)
    answers = batch_answer(args.question, passages, top_n=args.top_k)
    if args.explain and answers:
        exp = explain_qa(args.question, answers[0]["passage"])
        answers[0]["explanation"] = exp
    if getattr(args, "pretty", False) and _RICH_AVAILABLE:
        _print_colored_answers(args.question, answers)
    else:
        print(json.dumps({"question": args.question, "answers": answers}, indent=2))


def _is_valid_span(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    # Treat [CLS]/[SEP] or single punctuation as invalid "no-answer" spans
    invalid_markers = {"[CLS]", "[SEP]"}
    if t.upper() in invalid_markers:
        return False
    if len(t) < 3:
        return False
    return True


def _choose_highlight_index(answers) -> int:
    # Prefer the first non-empty, non-[CLS] span; fallback to 0
    for i, a in enumerate(answers):
        if _is_valid_span((a.get("answer") or "").strip()):
            return i
    return 0


def _print_colored_answers(question: str, answers):
    """Print answers with the top answer highlighted in yellow and others in white."""
    console = Console()
    console.print(Text("Question: ", style="bold") + Text(question))
    if not answers:
        console.print(Text("No answers found.", style="bold red"))
        return
    highlight_idx = _choose_highlight_index(answers)
    for i, a in enumerate(answers):
        prefix = f"{i+1}. "
        ans_text = (a.get("answer") or "").strip() or "<no answer>"
        score = a.get("score")
        # Build line with score hint
        line = Text(prefix)
        line.append(ans_text)
        if score is not None:
            line.append(f"  (score={score:.3f})", style="dim")
        if i == highlight_idx:
            # Highlight top answer in yellow & bold
            line.stylize("bold yellow")
        else:
            line.stylize("white")
        console.print(line)


def cmd_recommend(args):
    recs = recommend_lawyers(args.query)
    print(json.dumps({"query": args.query, "recommendations": recs}, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Legal BERT QA & Lawyer Recommendation (XAI)")
    sub = ap.add_subparsers(dest="cmd")

    ask = sub.add_parser("ask", help="Ask a legal question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--explain", action="store_true", help="Add explanation for top answer")
    ask.add_argument("--pretty", action="store_true", help="Colorized output: top answer in yellow, others in white")
    ask.set_defaults(func=cmd_ask)

    rec = sub.add_parser("recommend", help="Recommend lawyer types for a query")
    rec.add_argument("query")
    rec.set_defaults(func=cmd_recommend)

    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()
