"""Decoder-only GPT-style Transformer LM.

Pipeline per forward:
    tokens -> token embeddings (+ learned pos OR RoPE in attention)
           -> N x [Norm -> Causal MHA -> residual -> Norm -> SwiGLU FFN -> residual]
           -> final Norm -> LM head -> logits
Loss: causal cross-entropy, shift(input)->target, ignore_index=-100 (for assistant-only masking / padding).

No retrieval, no lookup tables of answers: pure next-token prediction.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype=torch.float32):
    """Precompute RoPE cos/sin tables of shape [seq_len, head_dim]."""
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, hd]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D]; cos/sin: [T, D] -> broadcast."""
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)  # [1,1,T,D]
    sin = sin[:T].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.num_heads
        self.n_kv = cfg.num_kv_heads
        self.hd = cfg.head_dim
        self.n_rep = self.n_head // self.n_kv
        d = cfg.hidden_size
        self.q_proj = nn.Linear(d, self.n_head * self.hd, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos=None, sin=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.hd).transpose(1, 2)  # [B,H,T,D]
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        if cos is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        if self.n_rep > 1:  # GQA expand
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        # fused causal attention when available, else manual mask
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop.p if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.drop(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.hidden_size, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.hidden_size, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        norm = RMSNorm if cfg.norm == "rmsnorm" else nn.LayerNorm
        self.ln1 = norm(cfg.hidden_size)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = norm(cfg.hidden_size)
        self.mlp = SwiGLUFFN(cfg)

    def forward(self, x, cos=None, sin=None):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.hidden_size) if cfg.pos_encoding == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        norm = RMSNorm if cfg.norm == "rmsnorm" else nn.LayerNorm
        self.ln_f = norm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids, labels=None, attention_mask=None):
        B, T = input_ids.shape
        assert T <= self.cfg.context_length, f"seq len {T} > context {self.cfg.context_length}"
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_emb(pos)
        x = self.drop(x)
        cos = sin = None
        if self.cfg.pos_encoding == "rope":
            cos, sin = build_rope_cache(T, self.cfg.head_dim, self.cfg.rope_theta, input_ids.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B,T,V]
        loss = None
        if labels is not None:
            # causal shift: predict token t+1 from prefix ..t
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.cfg.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, path, extra: dict | None = None):
        from pathlib import Path
        import json
        import safetensors.torch as st
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.cfg.save(path / "config.json")
        st.save_file({k: v.clone() for k, v in self.state_dict().items()}, path / "model.safetensors")
        if extra:
            with open(path / "train_meta.json", "w", encoding="utf-8") as f:
                json.dump(extra, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_pretrained(cls, path, device="cpu") -> "MiniGPT":
        from pathlib import Path
        import safetensors.torch as st
        path = Path(path)
        cfg = ModelConfig.load(path / "config.json")
        # backward-compat: checkpoint config.json may contain only model fields
        model = cls(cfg)
        sd = st.load_file(path / "model.safetensors", device=device)
        model.load_state_dict(sd, strict=True)
        return model.to(device)
