from __future__ import annotations
import os
import json
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple
import asyncio
import time

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
from urllib.parse import urlparse, parse_qs, unquote, quote_plus

from .data_loading import load_corpora, infer_practice_areas
from .retrieval import build_corpus_index, retrieve_top_k
from .qa import batch_answer
from .explain import explain_qa
from .recommend import recommend_lawyers
from .generation import synthesize_answer

# Globals prepared at startup
APP_INDEX_LIMIT = int(os.environ.get("LEGAL_XAI_INDEX_LIMIT", os.environ.get("API_INDEX_LIMIT", "200")))
INDEX = None

app = FastAPI(title="Legal BERT XAI API", version="0.1.0")


def _init_index_once():
    global INDEX
    if INDEX is None:
        df = load_corpora(limit=APP_INDEX_LIMIT)
        INDEX = build_corpus_index(df)
    return INDEX


@app.on_event("startup")
async def _startup():
    # Build index in a thread to keep event loop free
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _init_index_once)


@app.get("/health")
async def health():
    return {"status": "ok", "index_size": len(INDEX.ids) if INDEX else 0}


def _is_valid_span(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    if t.upper() in {"[CLS]", "[SEP]"}:
        return False
    if len(t) < 3:
        return False
    return True


def _clean_answer_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("[CLS]", "").replace("[SEP]", "")
    t = " ".join(t.split())
    return t.strip(" .,")


def rerank_preferring_spans(answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Penalize CLS/no-answer by subtracting a margin
    MARGIN = 5.0
    scored = []
    for a in answers:
        s = float(a.get("score", 0.0))
        if not _is_valid_span(a.get("answer", "")):
            s = s - MARGIN
        scored.append((s, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in scored]


UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    )
}


def _scrape_indiankanoon(query: str, n: int, timeout: float) -> List[Tuple[str, str]]:
    # Domain-specific results; less likely to be blocked
    try:
        params = {"formInput": query}
        r = requests.get("https://indiankanoon.org/search/", params=params, headers=UA_HEADERS, timeout=timeout)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out: List[Tuple[str, str]] = []
        for a in soup.select(".result_title a"):
            href = a.get("href")
            title = a.get_text(strip=True) or href
            if href and href.startswith("/"):
                href = "https://indiankanoon.org" + href
            if href:
                out.append((href, title))
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def _scrape_duckduckgo(query: str, n: int, timeout: float) -> List[Tuple[str, str]]:
    try:
        params = {"q": query, "kl": "in-en"}  # India English locale
        r = requests.get("https://duckduckgo.com/html/", params=params, headers=UA_HEADERS, timeout=timeout)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out: List[Tuple[str, str]] = []
        for a in soup.select("a.result__a"):
            href = a.get("href")
            title = a.get_text(strip=True) or href
            if href:
                out.append((href, title))
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def _scrape_bing(query: str, n: int, timeout: float) -> List[Tuple[str, str]]:
    try:
        params = {"q": query}
        r = requests.get("https://www.bing.com/search", params=params, headers=UA_HEADERS, timeout=timeout)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out: List[Tuple[str, str]] = []
        for a in soup.select("li.b_algo h2 a"):
            href = a.get("href")
            title = a.get_text(strip=True) or href
            if href:
                out.append((href, title))
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def fetch_web_links(query: str, n: int = 5, timeout: float = 10.0) -> List[Dict[str, str]]:
    """Try multiple sources to fetch links reliably."""
    query = bias_india_query(query)
    tuples: List[Tuple[str, str]] = []
    for fn in (_scrape_indiankanoon, _scrape_duckduckgo, _scrape_bing):
        if len(tuples) >= n:
            break
        res = fn(query, n, timeout)
        for t in res:
            if t not in tuples:
                tuples.append(t)
            if len(tuples) >= n:
                break
    items: List[Dict[str, str]] = []
    for i, (url, title) in enumerate(tuples[:n], start=1):
        items.append({
            "label": f"link-{i}",
            "type": "link",
            "url": url,
            "text": title,
        })
    return filter_india_links(items, desired=n)


def build_explanation_text(explanation: Dict[str, Any], max_tokens: int = 15) -> str:
    try:
        ctx_tokens = explanation.get("context_tokens", [])
        ctx_scores = explanation.get("context_importances", [])
        if not ctx_tokens or not ctx_scores:
            return ""
        pairs = list(zip(ctx_tokens, ctx_scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top = [tok for tok, _ in pairs[:max_tokens] if tok.isalpha()]
        if not top:
            top = [tok for tok, _ in pairs[:max_tokens]]
        return "Key context tokens influencing the answer: " + ", ".join(top)
    except Exception:
        return ""


def decode_ddg_url(url: str) -> str:
    """Decode DuckDuckGo redirect URLs like //duckduckgo.com/l/?uddg=<target> to direct links."""
    try:
        if url.startswith("//duckduckgo.com/l/"):
            parsed = urlparse("https:" + url)
            qs = parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return unquote(qs["uddg"][0])
        if url.startswith("//"):
            return "https:" + url
        return url
    except Exception:
        return url


INDIA_HOST_HINTS = [
    ".gov.in", ".nic.in", ".ac.in", ".in/", ".in?", ".in#",
    "indiankanoon.org", "constitutionofindia.net", "legislative.gov.in",
    "lawmin.gov.in", "prsindia.org", "supremecourtofindia.nic.in", "districts.ecourts.gov.in",
]


def is_india_link(u: str) -> bool:
    ul = u.lower()
    return any(h in ul for h in INDIA_HOST_HINTS)


def filter_india_links(items: List[Dict[str, str]], desired: int) -> List[Dict[str, str]]:
    # Prefer India-oriented links but fall back to original results to avoid empty lists
    primary = [it for it in items if is_india_link(it.get("url", ""))]
    if not primary:
        return items[:desired]
    if len(primary) >= desired:
        return primary[:desired]
    rest = [it for it in items if it not in primary]
    return (primary + rest)[:desired]


def bias_india_query(q: str) -> str:
    # Encourage Indian legal context in web queries
    low = q.lower()
    if any(tok in low for tok in ["india", "indian", "ipc", "crpc", "article", "constitution", "supreme court", "high court", "act", "section"]):
        # Strengthen with 'law' if not present
        if "law" not in low:
            return q + " law"
        return q
    # Generic queries: append stronger India legal context
    return q + " India law"


RELATED_TEMPLATES = {
    "constitutional": [
        "What is Article 19 about?",
        "What are reasonable restrictions under Article 19?",
        "What remedies are available for violation of fundamental rights?",
    ],
    "criminal": [
        "Can police arrest without warrant?",
        "What are bailable vs non-bailable offences?",
        "What is anticipatory bail under CrPC?",
    ],
    "civil": [
        "What is a plaint and how is it filed?",
        "What is an injunction and when is it granted?",
        "How does execution of a decree work?",
    ],
}


def suggest_related_questions(query: str, top_answer: str, areas: List[str]) -> List[str]:
    qs: List[str] = []
    ql = query.lower()
    # Article-specific suggestions
    import re as _re
    m = _re.search(r"article\s+(\d+\w?)", ql)
    if m:
        art = m.group(1)
        qs.extend([
            f"What is Article {art}?",
            f"What is the scope of Article {art}?",
            f"What are the exceptions to Article {art}?",
        ])
    # Area-based suggestions
    for a in areas:
        for s in RELATED_TEMPLATES.get(a, []):
            if s not in qs:
                qs.append(s)
    # Generic fallbacks
    if not qs:
        qs = [
            "What other laws are relevant to this?",
            "What are common exceptions or defenses?",
            "Which court has jurisdiction in such matters?",
        ]
    return qs[:5]


def fallback_explanation(question: str, answer: str, passage: str, max_words: int = 30) -> str:
    if not passage:
        return ""
    words = passage.split()
    snippet = " ".join(words[:max_words])
    return f"Derived from retrieved context: {snippet}…"


LEGAL_KEYWORDS = [
    "article", "constitution", "ipc", "crpc", "writ", "petition", "bail", "arrest", "fir",
    "fundamental right", "habeas", "mandamus", "injunction", "plaint", "suit", "decree",
    # Added broader judiciary and statute cues
    "supreme court", "high court", "court", "case", "case law", "judge", "tribunal", "act",
    "section", "penal code", "evidence act", "contract act", "rt i", "public interest litigation",
]


def is_legal_query(text: str) -> bool:
    tl = text.lower()
    return any(k in tl for k in LEGAL_KEYWORDS)


def sse_format(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _max_passage_score(passages: List[Dict[str, Any]]) -> float:
    try:
        return max(float(p.get("score") or 0.0) for p in passages) if passages else 0.0
    except Exception:
        return 0.0


def is_out_of_scope(q: str, passages: List[Dict[str, Any]], threshold: float = 0.35) -> bool:
    """Heuristic: out-of-legal-context if query lacks legal cues AND similarity too low.

    If a high-priority legal keyword is present, treat as in-scope regardless of similarity.
    """
    ql = q.lower()
    HIGH_PRIORITY = [
        "supreme court", "high court", "ipc", "crpc", "article", "constitution",
        "fundamental rights", "writ", "bail", "fir", "tribunal", "act", "section",
    ]
    if any(k in ql for k in HIGH_PRIORITY):
        return False
    has_legal_cues = is_legal_query(q)
    top_sim = _max_passage_score(passages)
    # For very short queries, lower the similarity threshold slightly
    tok_count = len(ql.split())
    adj_threshold = threshold - 0.1 if tok_count <= 2 else threshold
    return (not has_legal_cues) and (top_sim < adj_threshold)


def _fallback_search_links(q: str, desired: int) -> List[Dict[str, str]]:
    bq = bias_india_query(q)
    enc = quote_plus(bq)
    candidates = [
        (f"https://www.google.com/search?q={enc}", "Google search (India-biased)"),
        (f"https://duckduckgo.com/?q={enc}", "DuckDuckGo search (India-biased)"),
        (f"https://www.bing.com/search?q={enc}", "Bing search (India-biased)"),
    ]
    out: List[Dict[str, str]] = []
    for i, (u, title) in enumerate(candidates[:max(1, desired)], start=1):
        out.append({"label": f"link-{i}", "type": "link", "url": u, "text": title})
    return out


def ensure_links(q: str, items: List[Dict[str, str]], desired: int) -> List[Dict[str, str]]:
    if items:
        return items
    # Provide generic search links as a last resort
    return _fallback_search_links(q, desired)


@app.get("/ask")
async def ask(
    q: str = Query(..., description="Question"),
    top_k: int = Query(3, ge=1, le=10),
    explain: bool = Query(False),
    stream: bool = Query(True),
    links: int = Query(5, ge=0, le=10),
    synthesize: bool = Query(False, description="Return a synthesized explanatory answer with citations"),
):
    index = _init_index_once()

    async def _gen() -> AsyncGenerator[bytes, None]:
        # retrieval
        passages = retrieve_top_k(index, q, k=top_k)
        # Out-of-legal-context check (relaxed): lacks legal cues AND low similarity
        if is_out_of_scope(q, passages, threshold=0.35):
            # still provide helpful links (India-biased) even if out-of-context
            web: List[Dict[str, str]] = []
            if links:
                try:
                    from tests.scrap_test import get_top_links  # type: ignore
                    res = get_top_links(bias_india_query(q), n=links)
                    web = [
                        {"label": f"link-{i+1}", "type": "link", "url": decode_ddg_url(item.get("link", "")), "text": item.get("title", "")}
                        for i, item in enumerate(res)
                        if item.get("link")
                    ]
                except Exception:
                    web = fetch_web_links(q, n=links)
            web = filter_india_links(web, desired=links or len(web))
            web = ensure_links(q, web, desired=links or 5)
            final_payload = {
                "top-answer": "Out of legal context for this assistant. Please ask a legal question related to Indian law.",
                "explaination-text": "",
                "links": web,
                "related-topics": "",
                "related-questions": [
                    "What is Article 14 about?",
                    "How to file an FIR in India?",
                    "What are the types of writs under the Constitution?",
                ],
            }
            yield sse_format({"type": "final", "data": final_payload}).encode("utf-8")
            yield sse_format({"type": "done"}).encode("utf-8")
            return
        yield sse_format({"type": "retrieval", "passages": passages}).encode("utf-8")
        await asyncio.sleep(0)  # yield control

        # qa
        answers = batch_answer(q, passages, top_n=top_k)
        answers = rerank_preferring_spans(answers)
        yield sse_format({"type": "answers", "answers": answers}).encode("utf-8")
        await asyncio.sleep(0)

        # web links: prefer test scraper if available
        web: List[Dict[str, str]] = []
        if links:
            try:
                from tests.scrap_test import get_top_links  # type: ignore
                res = get_top_links(bias_india_query(q), n=links)
                web = [
                    {"label": f"link-{i+1}", "type": "link", "url": decode_ddg_url(item.get("link", "")), "text": item.get("title", "")}
                    for i, item in enumerate(res)
                    if item.get("link")
                ]
            except Exception:
                web = fetch_web_links(q, n=links)
        web = filter_india_links(web, desired=links or len(web))
        web = ensure_links(q, web, desired=links or 5)
        yield sse_format({"type": "web", "links": web}).encode("utf-8")
        await asyncio.sleep(0)

        # explanation (top answer only)
        exp_text = ""
        if explain and answers:
            try:
                exp = explain_qa(q, answers[0]["passage"])
                exp_text = build_explanation_text(exp)
                if not exp_text:
                    exp_text = fallback_explanation(q, answers[0]["answer"], answers[0]["passage"]) or "Model attributions unavailable; explanation based on retrieved context."
                yield sse_format({"type": "explanation", "explanation": exp, "explaination-text": exp_text}).encode("utf-8")
            except Exception as e:
                yield sse_format({"type": "explanation_error", "error": str(e)}).encode("utf-8")
        # synthesis (optional)
        synthesis_payload = None
        if synthesize and passages:
            try:
                synthesis_payload = synthesize_answer(q, passages, style="detailed")
                yield sse_format({"type": "synthesis", "synthesized": synthesis_payload}).encode("utf-8")
            except Exception as e:
                yield sse_format({"type": "synthesis_error", "error": str(e)}).encode("utf-8")
        # final shaped payload
        top_answer = answers[0]["answer"] if answers else ""
        if not _is_valid_span(top_answer) and answers:
            # fallback to next
            for a in answers[1:]:
                if _is_valid_span(a.get("answer", "")):
                    top_answer = a["answer"]
                    break
        top_answer = _clean_answer_text(top_answer)
        if (not top_answer) and passages:
            # Fallback to context snippet in Indian context
            top_answer = " ".join(passages[0]["text"].split()[:40]) + "…"
        areas = infer_practice_areas(q)
        related = ", ".join(areas)
        related_qs = suggest_related_questions(q, top_answer, areas)
        final_payload = {
            "top-answer": top_answer,
            "explaination-text": exp_text if (explain and answers) else "",
            "links": web,
            "related-topics": related,
            "related-questions": related_qs,
            "synthesized-answer": synthesis_payload.get("text") if synthesis_payload else "",
            "synthesized-citations": synthesis_payload.get("citations") if synthesis_payload else [],
            "synthesized-low-confidence": synthesis_payload.get("low_confidence") if synthesis_payload else False,
            "disclaimer": "This response is generated from retrieved legal context; not a substitute for professional legal advice.",
        }
        yield sse_format({"type": "final", "data": final_payload}).encode("utf-8")
        yield sse_format({"type": "done"}).encode("utf-8")

    if stream:
        return StreamingResponse(_gen(), media_type="text/event-stream")

    # non-streaming JSON response
    passages = retrieve_top_k(index, q, k=top_k)
    if is_out_of_scope(q, passages, threshold=0.35):
        # Provide links even when out-of-context
        web: List[Dict[str, str]] = []
        if links:
            try:
                from tests.scrap_test import get_top_links  # type: ignore
                res = get_top_links(bias_india_query(q), n=links)
                web = [
                    {"label": f"link-{i+1}", "type": "link", "url": decode_ddg_url(item.get("link", "")), "text": item.get("title", "")}
                    for i, item in enumerate(res)
                    if item.get("link")
                ]
            except Exception:
                web = fetch_web_links(q, n=links)
        web = filter_india_links(web, desired=links or len(web))
        web = ensure_links(q, web, desired=links or 5)
        areas = []
        final_payload = {
            "top-answer": "Out of legal context for this assistant. Please ask a legal question related to Indian law.",
            "explaination-text": "",
            "links": web,
            "related-topics": "",
            "related-questions": [
                "What is Article 14 about?",
                "How to file an FIR in India?",
                "What are the types of writs under the Constitution?",
            ],
        }
        return JSONResponse(final_payload)
    answers = batch_answer(q, passages, top_n=top_k)
    synthesis_payload = None
    if synthesize and passages:
        try:
            synthesis_payload = synthesize_answer(q, passages, style="detailed")
        except Exception:
            synthesis_payload = None
    answers = rerank_preferring_spans(answers)
    # web links: prefer test scraper if available
    web: List[Dict[str, str]] = []
    if links:
        try:
            from tests.scrap_test import get_top_links  # type: ignore
            res = get_top_links(bias_india_query(q), n=links)
            web = [
                {"label": f"link-{i+1}", "type": "link", "url": decode_ddg_url(item.get("link", "")), "text": item.get("title", "")}
                for i, item in enumerate(res)
                if item.get("link")
            ]
        except Exception:
            web = fetch_web_links(q, n=links)
    web = filter_india_links(web, desired=links or len(web))
    web = ensure_links(q, web, desired=links or 5)
    top_answer = answers[0]["answer"] if answers else ""
    if not _is_valid_span(top_answer) and answers:
        for a in answers[1:]:
            if _is_valid_span(a.get("answer", "")):
                top_answer = a["answer"]
                break
    top_answer = _clean_answer_text(top_answer)
    if (not top_answer) and passages:
        top_answer = " ".join(passages[0]["text"].split()[:40]) + "…"
    exp_text = ""
    if explain and answers:
        try:
            exp = explain_qa(q, answers[0]["passage"])
            exp_text = build_explanation_text(exp)
            if not exp_text:
                exp_text = fallback_explanation(q, top_answer, answers[0]["passage"]) or "Model attributions unavailable; explanation based on retrieved context."
        except Exception:
            exp_text = ""
    areas = infer_practice_areas(q)
    related = ", ".join(areas)
    related_qs = suggest_related_questions(q, top_answer, areas)
    final_payload = {
        "top-answer": top_answer,
         "synthesized-answer": synthesis_payload.get("text") if synthesis_payload else "",
        "explaination-text": exp_text,
        "links": web,
        "related-topics": related,
        "related-questions": related_qs,
       
        "synthesized-citations": synthesis_payload.get("citations") if synthesis_payload else [],
        "synthesized-low-confidence": synthesis_payload.get("low_confidence") if synthesis_payload else False,
        "disclaimer": "This response is generated from retrieved legal context; not a substitute for professional legal advice.",
    }
    return JSONResponse(final_payload)


@app.get("/recommend")
async def recommend(
    q: str = Query(..., description="Query describing the legal situation"),
    top_k: int = Query(3, ge=1, le=6),
    areas_only: bool = Query(False, description="Return only area labels without keyword details"),
):
    """Recommend lawyer types / practice areas for a query.

    Response shape:
    {
      "query": str,
      "recommendations": [ {area,label,keywords,score,confidence} ],
      "margin": float,
      "clarification_needed": bool,
      "clarifying-questions": [str],
      "related-questions": [str]
    }
    """
    result = recommend_lawyers(q, top_k=top_k)
    recs = result.get("recommendations", [])
    if areas_only:
        recs = [{"area": r.get("area"), "label": r.get("label")} for r in recs]
    areas = [r.get("area") for r in result.get("recommendations", [])]
    related_qs = suggest_related_questions(q, "", [a for a in areas if a])
    payload = {
        "query": q,
        "recommendations": recs,
        "margin": result.get("margin", 0.0),
        "clarification_needed": result.get("clarification_needed", False),
        "clarifying-questions": result.get("clarifying_questions", []),
        "related-questions": related_qs,
    }
    return JSONResponse(payload)

@app.get("/")
async def root():
    return {"message": "Legal BERT XAI API. Use GET /ask?q=...&stream=true or /recommend?q=..."}
