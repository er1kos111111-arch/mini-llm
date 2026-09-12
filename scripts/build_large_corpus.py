"""Build a large-scale pretraining corpus: Wikipedia RU+EN + local corpus.

Why: 5k lines memorises; millions of lines generalise. This is the single
biggest lever toward Qwen-like behaviour at our scale.

Sources (in order of preference):
  1. HuggingFace `datasets` Wikipedia dumps (needs `datasets` pip package):
       wikipedia, 20220301.ru / 20220301.en  (plain text field 'text')
  2. Fallback: local files (data/pretrain_corpus.txt + conversations).

Output: data/large/pretrain_large.txt (one article-chunk per line block).

Usage (Colab, datasets>=2.16, ~10-30 GB disk for full dump):
    python scripts/build_large_corpus.py --ru-articles 200000 --en-articles 200000
Quick smoke (no download, subsample-friendly):
    python scripts/build_large_corpus.py --ru-articles 2000 --en-articles 2000

Each article is split into ~300-word paragraphs; empties/headers dropped.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

OUT = Path("data/large/pretrain_large.txt")
SEED = 7


def clean_paragraphs(text: str) -> list[str]:
    out = []
    for para in text.split("\n"):
        p = " ".join(para.split())
        if len(p) < 120:  # drop headers/lists stubs
            continue
        if p.startswith(("=", "|", "{", "}", "<", "File:", "Файл:")):
            continue
        out.append(p[:2000])
    return out


def load_wiki(lang: str, n: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("wikipedia", f"20220301.{lang}", split="train", trust_remote_code=False)
    idx = list(range(min(n, len(ds))))
    rng = random.Random(SEED)
    rng.shuffle(idx)
    paras: list[str] = []
    for i in idx:
        paras.extend(clean_paragraphs(ds[int(i)]["text"]))
        if len(paras) >= n * 6:
            break
    return paras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ru-articles", type=int, default=20000)
    ap.add_argument("--en-articles", type=int, default=20000)
    ap.add_argument("--max-lines", type=int, default=0, help="0 = no cap")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    try:
        print("loading RU wikipedia...")
        lines += load_wiki("ru", args.ru_articles)
        print(f"ru paragraphs: {len(lines)}")
        print("loading EN wikipedia...")
        before = len(lines)
        lines += load_wiki("en", args.en_articles)
        print(f"en paragraphs: {len(lines) - before}")
    except Exception as e:
        print(f"[warn] wikipedia download failed ({e}); using local corpus only")

    # always mix in the local curated corpus (dialogues teach style)
    for f in ["data/pretrain_corpus.txt", "data/conversations_text.txt"]:
        p = Path(f)
        if p.exists():
            lines += [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(lines)
    if args.max_lines:
        lines = lines[: args.max_lines]
    with open(OUT, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    toks = sum(len(l.split()) for l in lines)
    print(f"wrote {len(lines)} lines (~{toks/1e6:.1f}M words) -> {OUT}")


if __name__ == "__main__":
    main()
