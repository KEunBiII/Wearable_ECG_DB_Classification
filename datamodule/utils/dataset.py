# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

DEFAULT_INTERVAL_NAMES = [
    "P_dur_ms", "PR_ms", "QRS_ms", "ST_ms", "T_dur_ms",
    "QT_ms", "QTcB_ms", "QTcF_ms", "QTcH_ms", "QTcFm_ms",
]


class MemmapRecordStore:
    """
    캐시 디렉터리의 .npy 파일을 memmap으로 열어 보관합니다.
    같은 경로를 다시 요청하면 이미 열린 파일을 그대로 반환합니다.
    (DataLoader worker 당 독립적인 인스턴스를 가집니다.)

    extra_npy : ReturnConfig.extra_npy와 동일한 파일명 리스트
    """

    def __init__(self, extra_npy: Optional[List[str]] = None):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._extra = extra_npy or []

    def get(self, cache_path: str) -> Dict[str, Any]:
        if cache_path in self._store:
            return self._store[cache_path]

        def _mmap(name: str):
            return np.load(os.path.join(cache_path, name), mmap_mode="r")

        d: Dict[str, Any] = {
            "X":         _mmap("X.npy"),
            "r_idx":     _mmap("r_idx.npy"),
            "seg_id":    _mmap("seg_id.npy"),
            "seg_start": _mmap("seg_start.npy"),
            "seg_len":   _mmap("seg_len.npy"),
            "fs":        float(_mmap("fs.npy")[0]),
        }

        for fname in ("y.npy", "Y.npy", "labels.npy"):
            p = os.path.join(cache_path, fname)
            if os.path.isfile(p):
                d["labels"] = np.load(p, mmap_mode="r")
                break

        p = os.path.join(cache_path, "intervals.npy")
        if os.path.isfile(p):
            d["intervals"] = np.load(p, mmap_mode="r")
            np_json = os.path.join(cache_path, "interval_names.json")
            if os.path.isfile(np_json):
                with open(np_json, "r", encoding="utf-8") as f:
                    d["interval_names"] = json.load(f)
            else:
                d["interval_names"] = DEFAULT_INTERVAL_NAMES

        p = os.path.join(cache_path, "meta.json")
        d["demo"] = json.load(open(p, "r", encoding="utf-8")) if os.path.isfile(p) else {}

        for fname in self._extra:
            p = os.path.join(cache_path, fname)
            key = fname[:-4] if fname.endswith(".npy") else fname
            if os.path.isfile(p):
                d[f"extra:{key}"] = np.load(p, mmap_mode="r")

        self._store[cache_path] = d
        return d
