"""Quality-gate evaluation: fixed RU/EN prompt set, printed after training."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.model import MiniGPT
from model.generation import generate

PROMPTS_RU = [
    "Привет",
    "Как дела?",
    "Расскажи о себе.",
    "Что такое Python?",
    "Как изучать программирование?",
    "Что такое искусственный интеллект?",
    "Объясни, что такое космос.",
]
PROMPTS_EN = [
    "Hello",
    "How are you?",
    "Tell me about yourself.",
    "What is Python?",
    "How can I learn programming?",
    "What is artificial intelligence?",
    "Explain what space is.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/tiny_run/export")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = Path(args.checkpoint)
    tok = Tokenizer.from_file(str(ck / "tokenizer.json" if (ck / "tokenizer.json").exists() else ck.parent / "tokenizer.json"))
    model = MiniGPT.load_pretrained(ck, device=device)
    model.eval()
    eos = tok.token_to_id("<EOS>")
    for lang, prompts in (("RU", PROMPTS_RU), ("EN", PROMPTS_EN)):
        print(f"===== {lang} =====")
        for p in prompts:
            prompt = f"<USER>\n{p}\n<ASSISTANT>\n"
            out = generate(model, tok, prompt, max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature, eos_token_id=eos, device=device)
            print(f"User: {p}\nAssistant: {out.strip()}\n")


if __name__ == "__main__":
    main()
