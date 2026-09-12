"""Padding collator: pads input_ids/labels/attention_mask to longest in batch."""
from __future__ import annotations

import torch


class CausalCollator:
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        maxlen = max(len(b["input_ids"]) for b in batch)
        inp, lab, msk = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            inp.append(torch.cat([b["input_ids"], torch.full((pad,), self.pad_id, dtype=torch.long)]))
            l = b["labels"]
            lab.append(torch.cat([l, torch.full((pad,), -100, dtype=torch.long)]))
            a = b.get("attention_mask", torch.ones(n, dtype=torch.long))
            msk.append(torch.cat([a, torch.zeros(pad, dtype=torch.long)]))
        return {"input_ids": torch.stack(inp), "labels": torch.stack(lab), "attention_mask": torch.stack(msk)}
