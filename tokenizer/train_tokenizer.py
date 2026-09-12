"""Train a Byte-Level BPE tokenizer on the project corpus.

Special tokens: <PAD> <UNK> <BOS> <EOS> <USER> <ASSISTANT>
Supports RU/EN/digits/punctuation via byte-level alphabet (no alphabet limit).

Usage:
    python -m tokenizer.train_tokenizer --config config/tiny.json
    python -m tokenizer.train_tokenizer --corpus data/pretrain_corpus.txt data/conversations_text.txt --vocab-size 8000 --out tokenizer/out_tiny
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

SPECIAL = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<USER>", "<ASSISTANT>"]


def train(corpus_files: list[str], vocab_size: int, out_dir: str, min_frequency: int = 2) -> Tokenizer:
    for f in corpus_files:
        if not Path(f).exists():
            raise FileNotFoundError(f"corpus file not found: {f}")
    tok = Tokenizer(BPE(unk_token="<UNK>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL,
        show_progress=True,
    )
    tok.train(corpus_files, trainer)
    # post-processor: wrap single sequence with BOS/EOS
    tok.post_processor = TemplateProcessing(
        single="<BOS> $A <EOS>",
        special_tokens=[("<BOS>", tok.token_to_id("<BOS>")), ("<EOS>", tok.token_to_id("<EOS>"))],
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    with open(out / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump({"special_tokens": SPECIAL, "vocab_size": vocab_size, "byte_level": True}, f, ensure_ascii=False, indent=2)
    return tok


def smoke_test(tok: Tokenizer):
    for s in [
        "Привет! Как дела?",
        "Hello! How are you?",
        "Я хочу изучать программирование.",
        "I want to learn programming.",
        "<USER> Привет <ASSISTANT> Привет! Чем помочь?",
    ]:
        enc = tok.encode(s)
        dec = tok.decode(enc.ids)
        print(f"IN : {s}\nIDS: {enc.ids[:32]}{'...' if len(enc.ids) > 32 else ''} (len={len(enc.ids)})\nOUT: {dec}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--corpus", nargs="*", default=None)
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--min-frequency", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = {}
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    t = cfg.get("tokenizer", {})
    corpus = args.corpus or t.get("corpus_files", ["data/pretrain_corpus.txt"])
    vocab_size = args.vocab_size or t.get("vocab_size", 8000)
    min_freq = args.min_frequency if args.min_frequency is not None else t.get("min_frequency", 2)
    out = args.out or t.get("output_dir", "tokenizer/out_tiny")

    tok = train(corpus, vocab_size, out, min_freq)
    print(f"saved to {out}/tokenizer.json vocab={tok.get_vocab_size()}")
    smoke_test(tok)


if __name__ == "__main__":
    main()
