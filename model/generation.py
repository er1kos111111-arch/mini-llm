"""Autoregressive generation: prompt -> tokens -> model -> next token -> append -> repeat.

Pure neural sampling. No retrieval / DB lookup anywhere.
Supports: temperature, top_k, top_p (nucleus), repetition_penalty, max_new_tokens, eos, greedy.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    for tok in set(input_ids.tolist()):
        if logits[tok] > 0:
            logits[tok] = logits[tok] / penalty
        else:
            logits[tok] = logits[tok] * penalty
    return logits


def _top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    out = logits.clone()
    V = out.shape[-1]
    if top_k and top_k > 0 and top_k < V:
        thr, _ = torch.topk(out, top_k)
        out[out < thr[..., -1, None]] = float("-inf")
    if top_p < 1.0:
        probs = F.softmax(out, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_idx_to_remove = torch.zeros_like(out, dtype=torch.bool).scatter_(-1, sorted_idx, mask)
        out[sorted_idx_to_remove] = float("-inf")
    return out


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    eos_token_id: int | None = None,
    do_sample: bool = True,
    device: str | None = None,
) -> str:
    """Generate continuation for a raw prompt string (already chat-templated by caller)."""
    model.eval()
    device = device or next(model.parameters()).device
    enc = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([enc.ids if hasattr(enc, "ids") else enc], dtype=torch.long, device=device)
    ctx = getattr(getattr(model, "cfg", None), "context_length", 512)
    if eos_token_id is None:
        eos_token_id = tokenizer.token_to_id("<EOS>")

    generated: list[int] = []
    for _ in range(max_new_tokens):
        cur = input_ids[:, -ctx:]  # sliding window if prompt too long
        out = model(cur)
        logits = out["logits"][0, -1]
        logits = _apply_repetition_penalty(logits.clone(), input_ids[0], repetition_penalty)
        if not do_sample or temperature <= 1e-6:
            nxt = int(torch.argmax(logits))
        else:
            logits = _top_k_top_p_filter(logits / max(temperature, 1e-6), top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
        input_ids = torch.cat([input_ids, torch.tensor([[nxt]], device=device)], dim=1)
        generated.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def generate_ids(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    eos_token_id: int | None = None,
    do_sample: bool = True,
) -> list[int]:
    """Low-level autoregressive loop over id tensors. Used by tests to prove step-by-step generation."""
    model.eval()
    device = next(model.parameters()).device
    cur = input_ids.to(device).clone()
    ctx = getattr(getattr(model, "cfg", None), "context_length", 512)
    out_ids: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(cur[:, -ctx:])["logits"][0, -1]
        logits = _apply_repetition_penalty(logits.clone(), cur[0], repetition_penalty)
        if not do_sample or temperature <= 1e-6:
            nxt = int(torch.argmax(logits))
        else:
            logits = _top_k_top_p_filter(logits / max(temperature, 1e-6), top_k, top_p)
            nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        out_ids.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
    return out_ids
