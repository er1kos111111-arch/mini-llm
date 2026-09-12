"""Misc helpers: seeding, param counting, config summary."""
from __future__ import annotations
import json, random
from pathlib import Path
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_config_summary(path: str = "config/config.json"):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    print(json.dumps(cfg, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    print_config_summary()
