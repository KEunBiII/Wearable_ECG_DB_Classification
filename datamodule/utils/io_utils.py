# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def hicardi_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tensor → torch.stack,  int/float → torch.tensor,  dict/str → 리스트 그대로.
    """
    out: Dict[str, Any] = {}
    for k in batch[0].keys():
        vals = [item[k] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            out[k] = torch.tensor(vals)
        else:
            out[k] = vals
    return out
