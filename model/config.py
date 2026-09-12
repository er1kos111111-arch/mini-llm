"""Model + training hyperparameter container."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    hidden_size: int = 384
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 8  # GQA; keep == num_heads for plain MHA
    ffn_mult: float = 4.0
    context_length: int = 512
    dropout: float = 0.0
    norm: str = "rmsnorm"          # "rmsnorm" | "layernorm"
    pos_encoding: str = "rope"     # "rope" | "learned"
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False

    # special token ids (filled after tokenizer training)
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        assert self.norm in ("rmsnorm", "layernorm")
        assert self.pos_encoding in ("rope", "learned")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def ffn_dim(self) -> int:
        return int(self.hidden_size * self.ffn_mult)

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    def num_params(self, verbose: bool = False) -> int:
        d, L, V = self.hidden_size, self.num_layers, self.vocab_size
        f = self.ffn_dim
        h, kv = self.num_heads, self.num_kv_heads
        hd = d // h
        # attn: q(d*hd*h) + k,v(d*hd*kv)*2 + o(d*d)
        attn = d * h * hd + 2 * d * kv * hd + d * d
        # ffn (SwiGLU has 3 matrices): 3*d*f
        ffn = 3 * d * f
        # norms: 2 per block * d
        norms = 2 * d
        block = attn + ffn + norms
        tok_emb = V * d
        out_head = 0 if self.tie_word_embeddings else V * d
        pos_emb = 0 if self.pos_encoding == "rope" else self.context_length * d
        total = tok_emb + L * block + d + out_head + pos_emb  # +d final norm
        if verbose:
            print(f"per-block ~{block/1e6:.2f}M x{L} = {L*block/1e6:.2f}M | "
                  f"tok_emb {tok_emb/1e6:.2f}M | head {out_head/1e6:.2f}M | total {total/1e6:.2f}M")
        return total
