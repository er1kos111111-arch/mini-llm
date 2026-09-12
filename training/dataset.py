"""Datasets: pretraining (plain text LM) + SFT (conversational, assistant-only loss)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

USER_T = "<USER>"
ASST_T = "<ASSISTANT>"
EOS_T = "<EOS>"
BOS_T = "<BOS>"


def format_chat(messages: list[dict]) -> str:
    """<USER>\n... \n<ASSISTANT>\n... [<USER>...] with EOS at end."""
    parts: list[str] = []
    for m in messages:
        tag = USER_T if m["role"] == "user" else ASST_T
        parts.append(f"{tag}\n{m['content'].strip()}\n")
    return "".join(parts) + EOS_T


class PretrainDataset(Dataset):
    """Plain causal-LM over lines of a text file, packed into fixed blocks."""

    def __init__(self, tok, corpus_path: str, max_seq_len: int = 512):
        text = Path(corpus_path).read_text(encoding="utf-8")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        # encode everything then chunk into blocks (sequence packing)
        ids: list[int] = []
        bos = tok.token_to_id(BOS_T)
        eos = tok.token_to_id(EOS_T)
        for ln in lines:
            enc = tok.encode(ln, add_special_tokens=False).ids
            ids += ([bos] if bos is not None else []) + enc + ([eos] if eos is not None else [])
        # pad to multiple of block size
        n_blocks = max(1, len(ids) // max_seq_len)
        ids = ids[: n_blocks * max_seq_len]
        self.blocks = [ids[i * max_seq_len:(i + 1) * max_seq_len] for i in range(n_blocks)]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        ids = torch.tensor(self.blocks[i], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


class SFTDataset(Dataset):
    """Conversational SFT. With assistant_only_loss=True, user tokens get label -100."""

    def __init__(self, tok, jsonl_path: str, max_seq_len: int = 512, assistant_only_loss: bool = True):
        self.tok = tok
        self.max_len = max_seq_len
        self.ao = assistant_only_loss
        self.rows: list[dict] = [json.loads(l) for l in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.u_ids = tok.encode(USER_T, add_special_tokens=False).ids
        self.a_ids = tok.encode(ASST_T, add_special_tokens=False).ids
        self.eos_id = tok.token_to_id(EOS_T)
        self.bos_id = tok.token_to_id(BOS_T)
        self.pad_id = tok.token_to_id("<PAD>")

    def __len__(self):
        return len(self.rows)

    def _encode_messages(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        ids: list[int] = []
        loss_mask: list[int] = []  # 1 = learn, 0 = ignore
        if self.bos_id is not None:
            ids.append(self.bos_id)
            loss_mask.append(0)
        for m in messages:
            tag_ids = self.u_ids if m["role"] == "user" else self.a_ids
            piece = tag_ids + self.tok.encode("\n" + m["content"].strip() + "\n", add_special_tokens=False).ids
            ids += piece
            # learn on assistant spans (tag + content); ignore user spans
            if m["role"] == "assistant":
                loss_mask += [1] * len(piece)
            else:
                loss_mask += [0] * len(piece)
        if self.eos_id is not None:
            ids.append(self.eos_id)
            loss_mask.append(1)
        if not self.ao:
            loss_mask = [1] * len(ids)
        return ids, loss_mask

    def __getitem__(self, i):
        ids, mask = self._encode_messages(self.rows[i]["messages"])
        ids = ids[: self.max_len]
        mask = mask[: self.max_len]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.tensor([t if m else -100 for t, m in zip(ids, mask)], dtype=torch.long)
        attn = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def split_rows(rows: list, val_split: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_val = max(1, int(len(rows) * val_split))
    val_idx = set(idx[:n_val])
    return [r for i, r in enumerate(rows) if i not in val_idx], [r for i, r in enumerate(rows) if i in val_idx]
