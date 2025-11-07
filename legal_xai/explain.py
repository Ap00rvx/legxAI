from __future__ import annotations
from typing import Dict, Any, List, Tuple
import torch
from .model_registry import get_qa_components
from captum.attr import IntegratedGradients


def _forward_for_target(inputs, target_pair: Tuple[int, int]):
    _, model = get_qa_components()
    outputs = model(**inputs)
    start_idx, end_idx = target_pair
    # select logits at target positions and sum into shape (1,1) for Captum
    start_logit = outputs.start_logits[0, start_idx]
    end_logit = outputs.end_logits[0, end_idx]
    return (start_logit + end_logit).view(1, 1)


def explain_qa(question: str, context: str, target: Tuple[int, int] | None = None, n_steps: int = 32) -> Dict[str, Any]:
    """Return token importances for the predicted (or provided) answer span.

    Output: { tokens, importances, context_span, context_tokens, context_importances }
    """
    tokenizer, model = get_qa_components()
    enc = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    # predict if target not provided
    with torch.no_grad():
        out = model(**enc)
    if target is None:
        s = int(torch.argmax(out.start_logits))
        e = int(torch.argmax(out.end_logits))
        if e < s:
            e = s
        target = (s, e)
    # Prepare inputs for embeddings attribution
    embeddings = model.get_input_embeddings()
    def forward_emb(inputs_embeds):
        return _forward_for_target({
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }, target)
    input_embeds = embeddings(input_ids)
    baseline = torch.zeros_like(input_embeds)
    ig = IntegratedGradients(forward_emb)
    # Output has shape (1,1); specify target=0 (class dim) for attribution
    attributions, _ = ig.attribute(inputs=input_embeds, baselines=baseline, target=0, return_convergence_delta=True, n_steps=n_steps)
    # aggregate attributions per token (L2 norm)
    token_importance = attributions.norm(p=2, dim=-1).squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # Identify context tokens span: [CLS] question [SEP] context [SEP]
    sep_positions = [i for i, t in enumerate(tokens) if t == tokenizer.sep_token]
    if len(sep_positions) >= 1:
        ctx_start = sep_positions[0] + 1
        ctx_end = len(tokens) - 1  # last [SEP]
    else:
        ctx_start, ctx_end = 0, len(tokens) - 1
    context_tokens = tokens[ctx_start:ctx_end]
    context_imps = token_importance[ctx_start:ctx_end]
    # normalize importances to 0..1 for readability
    if context_imps.numel() > 0:
        c = context_imps / (context_imps.max() + 1e-9)
        context_imps = c
    return {
        "tokens": tokens,
        "importances": token_importance.detach().cpu().tolist(),
        "context_tokens": context_tokens,
        "context_importances": context_imps.detach().cpu().tolist(),
        "target": target,
    }
