"""Train MiniGPT: stage 'pretrain' (plain LM) and/or stage 'sft' (conversational).

Features: AdamW, cosine/warmup scheduler, mixed precision (fp16/bf16/no),
gradient accumulation, grad clipping, checkpointing, resume, eval split,
best-checkpoint tracking, post-train sample generation.

Usage:
    python -m training.train --config config/tiny.json
    python -m training.train --config config/tiny.json --stage pretrain
    python -m training.train --config config/tiny.json --stage sft --resume checkpoints/tiny_run/last
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, random_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import ModelConfig
from model.model import MiniGPT
from training.dataset import PretrainDataset, SFTDataset
from training.collator import CausalCollator


def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_cfg(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_scheduler(opt, total_steps: int, warmup_ratio: float, kind: str):
    warmup = max(1, int(total_steps * warmup_ratio))
    if kind == "constant":
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    if kind == "linear":
        def fn(s):
            if s < warmup:
                return s / warmup
            return max(0.0, (total_steps - s) / max(1, total_steps - warmup))
        return torch.optim.lr_scheduler.LambdaLR(opt, fn)
    # cosine (default)
    def fn(s):
        if s < warmup:
            return s / warmup
        p = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        loss = model(b["input_ids"], labels=b["labels"])["loss"]
        tot += loss.item()
        n += 1
        if n >= 50:
            break
    model.train()
    return tot / max(1, n)


def save_ckpt(model, opt, sched, step, epoch, out: Path, cfg: dict, is_best=False):
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched else None,
        "step": step, "epoch": epoch,
    }, out / ("best.pt" if is_best else "last.pt"))
    # export HF-style dir alongside
    exp = out / "export"
    model.save_pretrained(exp, extra={"step": step, "epoch": epoch})


def train_stage(cfg: dict, stage: str, resume: str | None):
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} stage={stage}")

    tcfg, mcg, dcfg, hcfg = cfg["tokenizer"], cfg["model"], cfg["data"], cfg["train"]
    tok = Tokenizer.from_file(str(Path(tcfg["output_dir"]) / "tokenizer.json"))
    pad_id = tok.token_to_id("<PAD>")
    vocab = tok.get_vocab_size()
    if vocab != mcg["vocab_size"]:
        print(f"[warn] tokenizer vocab {vocab} != config vocab {mcg['vocab_size']}; using tokenizer size")
        mcg = dict(mcg, vocab_size=vocab)

    mcfg = ModelConfig(**{k: v for k, v in mcg.items() if k in ModelConfig.__dataclass_fields__},
                        pad_id=pad_id, unk_id=tok.token_to_id("<UNK>"),
                        bos_id=tok.token_to_id("<BOS>"), eos_id=tok.token_to_id("<EOS>"))
    print(f"params: {mcfg.num_params(verbose=True)/1e6:.2f}M")
    model = MiniGPT(mcfg).to(device)

    # dataset
    if stage == "pretrain":
        full = PretrainDataset(tok, dcfg["pretrain_corpus"], dcfg["max_seq_len"])
    else:
        full = SFTDataset(tok, dcfg["sft_file"], dcfg["max_seq_len"], dcfg.get("assistant_only_loss", True))
    n_val = max(1, int(len(full) * dcfg.get("val_split", 0.02)))
    n_tr = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_tr, n_val], generator=torch.Generator().manual_seed(cfg.get("seed", 42)))
    coll = CausalCollator(pad_id)
    train_loader = DataLoader(train_ds, batch_size=hcfg["batch_size"], shuffle=True,
                              num_workers=hcfg.get("num_workers", 0), collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=hcfg["batch_size"], collate_fn=coll)

    opt = torch.optim.AdamW(model.parameters(), lr=hcfg["learning_rate"],
                            weight_decay=hcfg.get("weight_decay", 0.01), betas=(0.9, 0.95))
    epochs = int(hcfg.get(f"{stage}_epochs", hcfg.get("epochs", 5)))
    total_steps = (len(train_loader) * epochs + hcfg["gradient_accumulation_steps"] - 1) // hcfg["gradient_accumulation_steps"]
    print(f"stage={stage} epochs={epochs} train_samples={n_tr} opt_steps={total_steps}")
    sched = get_scheduler(opt, total_steps, hcfg.get("warmup_ratio", 0.05), hcfg.get("scheduler", "cosine"))

    use_amp = hcfg.get("mixed_precision", "no") in ("fp16", "bf16") and device == "cuda"
    amp_dtype = torch.bfloat16 if hcfg.get("mixed_precision") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda") if (use_amp and amp_dtype == torch.float16) else None

    out_dir = Path(hcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    # copy tokenizer next to checkpoints for chat.py portability
    try:
        shutil.copy(Path(tcfg["output_dir"]) / "tokenizer.json", out_dir / "tokenizer.json")
    except OSError:
        pass

    step, best = 0, float("inf")
    if resume:
        rp = Path(resume)
        # resume from export dir (model.safetensors, weights only) or training ckpt (last.pt/best.pt)
        if rp.is_dir() and (rp / "model.safetensors").exists() and not (rp / "last.pt").exists():
            import safetensors.torch as st
            missing, unexpected = model.load_state_dict(st.load_file(rp / "model.safetensors", device=device), strict=False)
            print(f"resumed weights from export {resume} (missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            ck = torch.load(rp / "last.pt" if rp.is_dir() and (rp / "last.pt").exists() else rp, map_location=device)
            model.load_state_dict(ck["model"])
            try:
                opt.load_state_dict(ck["optimizer"])
                if ck.get("scheduler") and sched:
                    sched.load_state_dict(ck["scheduler"])
            except Exception as e:
                print(f"[warn] optimizer state not restored: {e}")
            step = ck.get("step", 0)
            print(f"resumed from {resume} at step {step}")

    accum = hcfg.get("gradient_accumulation_steps", 1)
    model.train()
    for epoch in range(epochs):
        for bi, b in enumerate(train_loader):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = model(b["input_ids"], labels=b["labels"])["loss"] / accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (bi + 1) % accum == 0:
                if scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), hcfg.get("max_grad_norm", 1.0))
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), hcfg.get("max_grad_norm", 1.0))
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % hcfg.get("log_steps", 20) == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item()*accum:.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
                if step % hcfg.get("eval_steps", 200) == 0:
                    vl = evaluate(model, val_loader, device)
                    print(f"  [eval] step {step} val_loss {vl:.4f} ppl {math.exp(min(vl, 20)):.1f}", flush=True)
                    if vl < best:
                        best = vl
                        save_ckpt(model, opt, sched, step, epoch, out_dir, cfg, is_best=True)
                        print("  [eval] new best -> best.pt", flush=True)
                if step % hcfg.get("save_steps", 400) == 0:
                    save_ckpt(model, opt, sched, step, epoch, out_dir, cfg)
        # end epoch eval
        vl = evaluate(model, val_loader, device)
        print(f"[epoch {epoch} done] val_loss {vl:.4f}", flush=True)
        save_ckpt(model, opt, sched, step, epoch, out_dir, cfg)
        if vl < best:
            best = vl
            save_ckpt(model, opt, sched, step, epoch, out_dir, cfg, is_best=True)

    print(f"training done. best val {best:.4f}. export in {out_dir/'export'}")
    # quick sample
    try:
        from model.generation import generate
        for p in ["<USER>\nПривет\n<ASSISTANT>\n", "<USER>\nWhat is Python?\n<ASSISTANT>\n"]:
            print(f"PROMPT: {p!r}\nSAMPLE: {generate(model, tok, p, max_new_tokens=48, temperature=0.8, device=str(device))}\n")
    except Exception as e:
        print(f"[warn] sample generation failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--stage", default=None, choices=["pretrain", "sft"])
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    stage = args.stage or cfg["train"].get("stage", "sft")
    resume = args.resume or cfg["train"].get("resume_from")
    train_stage(cfg, stage, resume)


if __name__ == "__main__":
    main()
