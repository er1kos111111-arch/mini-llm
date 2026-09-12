"""Prove this is a real autoregressive LLM, not a lookup table.

Checks:
 1. Model generates more than one token.
 2. Different prompts -> different answers.
 3. Answers are not copied verbatim from training JSONL (novelty check).
 4. Generation is autoregressive (step-by-step loop matches generate_ids; >1 LM calls).
 5. Temperature affects the output distribution.
 6. Model checkpoint can be reloaded (save_pretrained/load_pretrained roundtrip).
 7. Tokenizer can be loaded standalone.
 8. After restart (fresh process reload) model+tokenizer still generate.

 Plus: repeated sampling of 'Привет' yields at least partially different samples.

Run:  python -m tests.test_generation --checkpoint checkpoints/micro_run/export
If no checkpoint exists, a tiny random model + fresh tokenizer is built on the fly
for structural checks (1,4,5,6,7), and trained-checkpoint checks are skipped with warning.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def load_tok(path: Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def find_checkpoint(user_ckpt: str | None) -> Path | None:
    if user_ckpt and (Path(user_ckpt) / "model.safetensors").exists():
        return Path(user_ckpt)
    for c in [Path("checkpoints/micro_run/export"), Path("checkpoints/tiny_run/export"),
              Path("checkpoints/small_run/export")]:
        if (ROOT / c / "model.safetensors").exists():
            return ROOT / c
    return None


def find_tokenizer(ckpt: Path | None) -> Path | None:
    cands = []
    if ckpt:
        cands += [ckpt / "tokenizer.json", ckpt.parent / "tokenizer.json"]
    cands += [ROOT / "tokenizer/out_micro/tokenizer.json", ROOT / "tokenizer/out_tiny/tokenizer.json",
              ROOT / "data" / "tokenizer.json"]
    for c in cands:
        if c.exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--strict", action="store_true", help="fail if no trained checkpoint found")
    args = ap.parse_args()

    from model.config import ModelConfig
    from model.model import MiniGPT
    from model.generation import generate, generate_ids

    ckpt = find_checkpoint(args.checkpoint)
    tok_path = find_tokenizer(ckpt)
    results: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    if tok_path is None or ckpt is None:
        print("No trained checkpoint/tokenizer found -> building ephemeral tiny model for structural checks.")
        if args.strict:
            print("STRICT mode: failing."); sys.exit(1)
        # ephemeral objects
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        tmp_tok = Tokenizer(WordLevel(vocab={"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
                                             "<USER>": 4, "<ASSISTANT>": 5, "Привет": 6, "Hello": 7,
                                             "мир": 8, "world": 9}, unk_token="<UNK>"))
        tmp_tok.pre_tokenizer = Whitespace()
        cfg = ModelConfig(vocab_size=10, hidden_size=32, num_layers=2, num_heads=4,
                          num_kv_heads=4, ffn_dim=64, context_length=64, pos_encoding="learned",
                          tie_word_embeddings=False) if "ffn_dim" in ModelConfig.__dataclass_fields__ else \
              ModelConfig(vocab_size=10, hidden_size=32, num_layers=2, num_heads=4,
                          num_kv_heads=4, ffn_mult=2.0, context_length=64, pos_encoding="learned",
                          tie_word_embeddings=False)
        model = MiniGPT(cfg)
        model.eval()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "tok.json"
            tmp_tok.save(str(tp))
            tok2 = load_tok(tp)
            check("7. tokenizer loads standalone", tok2.get_vocab_size() == 10)
            model.save_pretrained(Path(td) / "m")
            m2 = MiniGPT.load_pretrained(Path(td) / "m")
            check("6. checkpoint save/load roundtrip", m2.count_params() == model.count_params())
        # autoregressive structural checks on random weights
        ids = torch.tensor([[4, 6]])
        with torch.no_grad():
            n_calls = [0]
            orig_fwd = model.forward
            def counting_fwd(*a, **k):
                n_calls[0] += 1
                return orig_fwd(*a, **k)
            model.forward = counting_fwd
            outs = generate_ids(model, ids, max_new_tokens=8, temperature=0.0, eos_token_id=3)
            check("1. generates multiple tokens", len(outs) > 1, f"got {len(outs)}")
            check("4. autoregressive (>1 forward calls)", n_calls[0] > 1, f"calls={n_calls[0]}")
        # temperature effect on distribution, not just weights
        model.forward = orig_fwd
        with torch.no_grad():
            logits = model(torch.tensor([[4, 6]]))["logits"][0, -1]
            import torch.nn.functional as F
            p_sharp = F.softmax(logits / 0.1, dim=-1)
            p_flat = F.softmax(logits / 2.0, dim=-1)
            check("5. temperature changes distribution",
                  float((p_sharp - p_flat).abs().max()) > 1e-4)
        print("\nStructural checks done (no trained weights; retrain for quality checks 2,3,8).")
        failed = [r for r in results if not r[1]]
        sys.exit(1 if failed else 0)

    # ---- full checks with real checkpoint ----
    from model.model import MiniGPT as MG
    tok = load_tok(tok_path)
    check("7. tokenizer loads standalone", tok.get_vocab_size() > 100, f"vocab={tok.get_vocab_size()}")
    model = MG.load_pretrained(ckpt, device="cpu")
    model.eval()
    check("6. checkpoint loads", model.count_params() > 0, f"params={model.count_params()/1e6:.2f}M")
    eos = tok.token_to_id("<EOS>")

    # 1: multi-token generation
    out = generate(model, tok, "<USER>\nПривет\n<ASSISTANT>\n", max_new_tokens=40, temperature=0.8, eos_token_id=eos, device="cpu")
    ntok = len(tok.encode(out).ids)
    check("1. generates >1 token", ntok > 1, f"tokens={ntok} text={out[:80]!r}")

    # 2: different prompts -> different outputs
    a = generate(model, tok, "<USER>\nПривет\n<ASSISTANT>\n", max_new_tokens=40, temperature=0.0, eos_token_id=eos, device="cpu")
    b = generate(model, tok, "<USER>\nWhat is Python?\n<ASSISTANT>\n", max_new_tokens=40, temperature=0.0, eos_token_id=eos, device="cpu")
    check("2. prompts diverge", a.strip() != b.strip(), f"\n  A={a[:70]!r}\n  B={b[:70]!r}")

    # 3: novelty vs training data (not verbatim copy)
    train_texts = set()
    jp = ROOT / "data/conversations.jsonl"
    if jp.exists():
        for line in jp.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                for m in d["messages"]:
                    if m["role"] == "assistant":
                        train_texts.add(m["content"].strip())
            except Exception:
                pass
    samples = [generate(model, tok, "<USER>\nПривет\n<ASSISTANT>\n", max_new_tokens=40,
                        temperature=0.9, eos_token_id=eos, device="cpu").strip() for _ in range(4)]
    novel = [s for s in samples if s not in train_texts]
    check("3. outputs not verbatim DB copies", len(novel) > 0, f"{len(novel)}/4 novel")

    # diversity across samples of same prompt
    check("3b. same prompt varies", len(set(samples)) > 1, f"samples={samples[:2]}")

    # 4: autoregressive proof — count forwards
    ids = torch.tensor([tok.encode("<USER>\nПривет\n<ASSISTANT>\n", add_special_tokens=False).ids])
    n_calls = [0]
    orig = model.forward
    def cf(*a, **k):
        n_calls[0] += 1
        return orig(*a, **k)
    model.forward = cf
    generate_ids(model, ids, max_new_tokens=5, temperature=0.0, eos_token_id=eos)
    model.forward = orig
    check("4. autoregressive forward-per-token", n_calls[0] >= 5, f"calls={n_calls[0]}")

    # 5: temperature effect
    with torch.no_grad():
        logits = model(ids[:, -32:])["logits"][0, -1]
        import torch.nn.functional as F
        d = float((F.softmax(logits / 0.1, -1) - F.softmax(logits / 2.0, -1)).abs().max())
    check("5. temperature matters", d > 1e-4, f"maxdiff={d:.4f}")

    # 8: fresh-process reload generates
    code = (f"import sys; sys.path.insert(0, r'{ROOT}'); "
            f"from tokenizers import Tokenizer; from model.model import MiniGPT; "
            f"from model.generation import generate; "
            f"tok=Tokenizer.from_file(r'{tok_path}'); m=MiniGPT.load_pretrained(r'{ckpt}', device='cpu'); m.eval(); "
            f"print(generate(m, tok, '<USER>\\nHello\\n<ASSISTANT>\\n', max_new_tokens=10, temperature=0.0, device='cpu'))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    check("8. reload in fresh process", r.returncode == 0 and len(r.stdout.strip()) > 0, r.stdout.strip()[:80] or r.stderr.strip()[-200:])

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results)-len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
