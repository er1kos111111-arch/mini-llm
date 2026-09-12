"""Interactive chat with a trained MiniGPT checkpoint.

Usage:
    python -m inference.chat --checkpoint checkpoints/tiny_run/export
    python -m inference.chat --checkpoint checkpoints/tiny_run/export --temperature 0.7 --top-p 0.9

Chat template:
    <USER>\n<text>\n<ASSISTANT>\n<reply><EOS> ...
The model continues autoregressively after the trailing <ASSISTANT>.
Type /quit to exit, /temp 0.7 to change temperature live.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.model import MiniGPT
from model.generation import generate


def load(checkpoint: str, device: str):
    ck = Path(checkpoint)
    tok_path = ck / "tokenizer.json"
    if not tok_path.exists():  # checkpoints/<run>/tokenizer.json copy, or tokenizer/out_*
        alt = ck.parent / "tokenizer.json"
        tok_path = alt if alt.exists() else tok_path
    tok = Tokenizer.from_file(str(tok_path))
    model = MiniGPT.load_pretrained(ck, device=device)
    model.eval()
    gen_cfg = {}
    meta = ck / "train_meta.json"
    return model, tok, gen_cfg


def build_prompt(history: list[tuple[str, str]], user_msg: str) -> str:
    parts = []
    for u, a in history:
        parts.append(f"<USER>\n{u.strip()}\n<ASSISTANT>\n{a.strip()}\n")
    parts.append(f"<USER>\n{user_msg.strip()}\n<ASSISTANT>\n")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/tiny_run/export")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, tok, _ = load(args.checkpoint, args.device)
    if args.tokenizer:
        tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<EOS>")
    print(f"Loaded {args.checkpoint} | params={model.count_params()/1e6:.1f}M | device={args.device}")
    print("Chat ( /quit to exit, /temp X to change temperature ).")
    history: list[tuple[str, str]] = []
    temp = args.temperature
    while True:
        try:
            user = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user.startswith("/temp"):
            try:
                temp = float(user.split()[1])
                print(f"temperature={temp}")
            except Exception:
                print("usage: /temp 0.7")
            continue
        prompt = build_prompt(history, user)
        # keep prompt within context
        ctx = model.cfg.context_length
        ids = tok.encode(prompt, add_special_tokens=False).ids
        if len(ids) > ctx - args.max_new_tokens:
            ids = ids[-(ctx - args.max_new_tokens):]
            prompt = tok.decode(ids)
        with torch.no_grad():
            reply = generate(model, tok, prompt, max_new_tokens=args.max_new_tokens,
                             temperature=temp, top_k=args.top_k, top_p=args.top_p,
                             repetition_penalty=args.repetition_penalty,
                             eos_token_id=eos, do_sample=temp > 0, device=args.device)
        print(f"\nAssistant: {reply.strip()}")
        history.append((user, reply.strip()))


if __name__ == "__main__":
    main()
