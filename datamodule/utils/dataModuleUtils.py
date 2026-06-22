import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 기존에 사용하던 프로젝트 내부 유틸리티 임포트
from .io_utils import load_json, save_json, hicardi_collate_fn
from .math_utils import (
    normalize as _normalize,
    compute_distances,
    DistanceMetric,
)
from .dataset import MemmapRecordStore, DEFAULT_INTERVAL_NAMES



DEFAULT_BEAT_CLASSES = [
    "Normal", "Sinus_Tachy", "APC", "AF_AFL",
    "Bradycardia", "VPC", "Trigeminy",
]

HICARDI_LABEL_COL_MAP: Dict[str, int] = {
    "Normal":      0,
    "VF_VT":       2,
    "VPC":         3,
    "Bigeminy":    5,
    "Trigeminy":   6,
    "Bradycardia": 8,
    "AF_AFL":      12,
    "APC":         14,
    "Sinus_Tachy": 16,
}

_NORMAL_COL    = HICARDI_LABEL_COL_MAP["Normal"]
_ABNORMAL_COLS = [v for k, v in HICARDI_LABEL_COL_MAP.items() if k != "Normal"]

OUT_OF_RANGE_CRITERIA: Dict[str, tuple] = {
    "PR_ms":   (120.0, 220.0),
    "QRS_ms":  (None,  120.0),
    "QTcB_ms": (350.0, 480.0),
    "QTcF_ms": (350.0, 480.0),
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Enum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SamplingStrategy(str, Enum):
    NONE           = "none"
    RANDOM         = "random"
    CENTROID_NEAR  = "centroid_near"
    CENTROID_FAR   = "centroid_far"
    DIST_THRESHOLD = "dist_threshold"


class PrototypingStrategy(str, Enum):
    RANDOM      = "random"
    CENTROID    = "centroid"
    HARD_MINING = "hard_mining"


class DatasetMode(str, Enum):
    BEAT     = "beat"
    SEQUENCE = "sequence"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Configs — 각 설정 dataclass는 자기 자신을 dict에서 파싱
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ReturnConfig:
    """각 샘플 dict에 어떤 필드를 넣을지."""
    waveform:    bool      = True
    labels:      bool      = False
    rr_interval: bool      = False
    hr:          bool      = False
    intervals:   bool      = False
    meta:        bool      = True
    demo:        bool      = False
    extra_npy:   List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ReturnConfig":
        d = dict(d or {})
        return cls(
            waveform    = bool(d.get("waveform",    True)),
            labels      = bool(d.get("labels",      False)),
            rr_interval = bool(d.get("rr_interval", False)),
            hr          = bool(d.get("hr",          False)),
            intervals   = bool(d.get("intervals",   False)),
            meta        = bool(d.get("meta",        True)),
            demo        = bool(d.get("demo",        False)),
            extra_npy   = list(d.get("extra_npy", []) or []),
        )


@dataclass
class DatasetConfig:
    """Dataset 구조 — mode, 정규화, 라벨 등 (return_arg 구조 키들)."""
    mode:          DatasetMode         = DatasetMode.BEAT
    seq_len:       int                 = 16
    stride:        int                 = 1
    normalize:     str                 = "z"
    window_half:   Optional[int]       = None
    label_names:   List[str]           = field(default_factory=lambda: list(DEFAULT_BEAT_CLASSES))
    n_classes:     int                 = 0
    cohort_filter: Optional[List[str]] = None
    min_beats:     int                 = 1
    cache_root:    Optional[str]       = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DatasetConfig":
        d = dict(d or {})
        mode = d.get("mode", "beat")
        if isinstance(mode, str):
            mode = DatasetMode(mode)
        return cls(
            mode          = mode,
            seq_len       = int(d.get("seq_len", 16)),
            stride        = int(d.get("stride", 1)),
            normalize     = d.get("normalize", "z"),
            window_half   = d.get("window_half", None),
            label_names   = list(d.get("label_names") or DEFAULT_BEAT_CLASSES),
            n_classes     = int(d.get("n_classes", 0)),
            cohort_filter = d.get("cohort_filter", None),
            min_beats     = int(d.get("min_beats", 1)),
            cache_root    = d.get("cache_root", None),
        )


@dataclass
class UndersamplingConfig:
    """Stage 1 — 정상/비정상 비율 조정 설정."""
    strategy:             SamplingStrategy = SamplingStrategy.NONE
    normal_ratio:         float            = 1.0
    distance_metric:      DistanceMetric   = DistanceMetric.EUCLIDEAN
    dist_threshold:       float            = 1.0
    keep_above_threshold: bool             = False
    centroid_max_samples: int              = 50_000
    seed:                 int              = 42
    apply_to_train:       bool             = True
    apply_to_val:         bool             = False
    apply_to_test:        bool             = False
    exclude_normal_col:   bool             = True

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "UndersamplingConfig":
        d = dict(d or {})
        strategy = d.get("strategy", "none")
        if isinstance(strategy, str):
            strategy = SamplingStrategy(strategy)
        metric = d.get("distance_metric", DistanceMetric.EUCLIDEAN)
        if isinstance(metric, str):
            metric = DistanceMetric(metric)
        return cls(
            strategy             = strategy,
            normal_ratio         = float(d.get("normal_ratio", 1.0)),
            distance_metric      = metric,
            dist_threshold       = float(d.get("dist_threshold", 1.0)),
            keep_above_threshold = bool(d.get("keep_above_threshold", False)),
            centroid_max_samples = int(d.get("centroid_max_samples", 50_000)),
            seed                 = int(d.get("seed", 42)),
            apply_to_train       = bool(d.get("apply_to_train", True)),
            apply_to_val         = bool(d.get("apply_to_val",   False)),
            apply_to_test        = bool(d.get("apply_to_test",  False)),
            exclude_normal_col   = bool(d.get("exclude_normal_col", True)),
        )

    def applies_to(self, split: str) -> bool:
        if self.strategy == SamplingStrategy.NONE:
            return False
        return {"train": self.apply_to_train,
                "val":   self.apply_to_val,
                "test":  self.apply_to_test}.get(split, False)


@dataclass
class MixupConfig:
    """Collate-time Mixup augmentation."""
    alpha:      float = 0.0
    apply_prob: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.alpha > 0.0 and self.apply_prob > 0.0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MixupConfig":
        d = dict(d or {})
        return cls(
            alpha      = float(d.get("alpha", 0.0)),
            apply_prob = float(d.get("apply_prob", 0.0)),
        )


@dataclass
class EfficiencyConfig:
    """Stage 2 — 빠른 프로토타이핑용 추가 서브샘플링 + Mixup."""
    apply_threshold:  int                 = 100_000
    variation_factor: float               = 1.0
    prototyping:      PrototypingStrategy = PrototypingStrategy.RANDOM
    mixup:            MixupConfig         = field(default_factory=MixupConfig)
    seed:             int                 = 42
    apply_to_train:   bool                = True
    apply_to_val:     bool                = False
    apply_to_test:    bool                = False

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EfficiencyConfig":
        d = dict(d or {})
        proto = d.get("prototyping", "random")
        if isinstance(proto, str):
            proto = PrototypingStrategy(proto)
        return cls(
            apply_threshold  = int(d.get("apply_threshold", 100_000)),
            variation_factor = float(d.get("variation_factor", 1.0)),
            prototyping      = proto,
            mixup            = MixupConfig.from_dict(d.get("mixup", {})),
            seed             = int(d.get("seed", 42)),
            apply_to_train   = bool(d.get("apply_to_train", True)),
            apply_to_val     = bool(d.get("apply_to_val",   False)),
            apply_to_test    = bool(d.get("apply_to_test",  False)),
        )

    def applies_to(self, split: str) -> bool:
        if self.variation_factor >= 1.0:
            return False
        return {"train": self.apply_to_train,
                "val":   self.apply_to_val,
                "test":  self.apply_to_test}.get(split, False)


@dataclass
class DataLoaderConfig:
    batch_size:  int  = 256
    num_workers: int  = 4
    pin_memory:  bool = True
    drop_last:   bool = True

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DataLoaderConfig":
        d = dict(d or {})
        return cls(
            batch_size  = int(d.get("batch_size", 256)),
            num_workers = int(d.get("num_workers", 4)),
            pin_memory  = bool(d.get("pin_memory", True)),
            drop_last   = bool(d.get("drop_last", True)),
        )


# 하위호환 alias
SamplingConfig = UndersamplingConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 데이터 객체 — Stage 결과 캡슐화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class BeatMask:
    """정상/비정상 beat flat 인덱스 (raw dataset 기준)."""
    normal:   List[int] = field(default_factory=list)
    abnormal: List[int] = field(default_factory=list)

    @property
    def all(self) -> List[int]: return self.normal + self.abnormal
    @property
    def n_normal(self)   -> int: return len(self.normal)
    @property
    def n_abnormal(self) -> int: return len(self.abnormal)
    @property
    def n_total(self)    -> int: return self.n_normal + self.n_abnormal
    @property
    def ratio(self) -> float:
        return self.n_normal / max(self.n_abnormal, 1)


@dataclass
class StageResult:
    """파이프라인 한 단계의 결과 — 항상 정확한 mask와 인덱스를 보유."""
    indices: List[int]    = field(default_factory=list)
    mask:    BeatMask     = field(default_factory=BeatMask)
    applied: bool         = False
    name:    str          = ""


@dataclass
class PipelineResult:
    """단일 split 파이프라인 실행 결과 — 모든 stage 정보 누적 보관."""
    raw_dataset:    Dataset
    initial_mask:   BeatMask
    stage1_result:  StageResult
    stage2_result:  StageResult
    active_dataset: Dataset

    @property
    def final_mask(self) -> BeatMask:
        return self.stage2_result.mask


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 원본 Dataset (기존 코드 유지)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HiCardiDataset(Dataset):
    """beat 단위 ECG Dataset."""

    def __init__(
        self,
        records:       List[Dict[str, str]],
        return_cfg:    ReturnConfig          = None,
        normalize:     str                   = "z",
        window_half:   Optional[int]         = None,
        label_names:   Optional[List[str]]   = None,
        n_classes:     int                   = 0,
        cohort_filter: Optional[List[str]]   = None,
        min_beats:     int                   = 1,
    ):
        self.return_cfg  = return_cfg or ReturnConfig()
        self.normalize   = normalize
        self.window_half = window_half
        self.label_names = label_names or DEFAULT_BEAT_CLASSES
        self.n_classes   = n_classes or len(self.label_names)

        if cohort_filter is not None:
            allowed = set(cohort_filter)
            records = [r for r in records if r.get("cohort", "") in allowed]
        self.records = [r for r in records if int(r.get("num_beats", 0)) >= min_beats]

        self.store = MemmapRecordStore(extra_npy=self.return_cfg.extra_npy)

        self.index: List[tuple] = [
            (ridx, b)
            for ridx, r in enumerate(self.records)
            for b in range(int(r.get("num_beats", 0)))
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ridx, beat_idx = self.index[idx]
        r   = self.records[ridx]
        rec = self.store.get(r["cache_path"])
        out: Dict[str, Any] = {}

        if self.return_cfg.waveform:
            x = np.asarray(rec["X"][beat_idx], dtype=np.float32)
            if self.window_half is not None:
                center = len(x) // 2
                s = max(0, center - self.window_half)
                e = min(len(x), center + self.window_half + 1)
                x = x[s:e]
            out["waveform"] = torch.from_numpy(_normalize(x, self.normalize))

        if self.return_cfg.labels:
            if "labels" in rec:
                lbl = np.asarray(rec["labels"][beat_idx], dtype=np.float32)
                if lbl.ndim == 0:
                    lbl = lbl.reshape(1)
                if lbl.shape[0] != self.n_classes:
                    cols = [HICARDI_LABEL_COL_MAP.get(n, i)
                            for i, n in enumerate(self.label_names)]
                    if all(c < lbl.shape[0] for c in cols):
                        lbl = lbl[cols]
                    else:
                        lbl = lbl[:self.n_classes]
                # all-zero → Normal (다른 라벨 없는 비트 = 정상)
                if lbl.sum() == 0 and "Normal" in self.label_names:
                    lbl[self.label_names.index("Normal")] = 1.0
            else:
                # sentinel -1: trainer가 해당 샘플을 필터링할 수 있도록 표시
                lbl = np.full(self.n_classes, -1.0, dtype=np.float32)
            out["labels"] = torch.from_numpy(lbl)

        rr_ms = 0.0
        if self.return_cfg.rr_interval or self.return_cfg.hr:
            if beat_idx > 0:
                rr_ms = (int(rec["r_idx"][beat_idx]) - int(rec["r_idx"][beat_idx - 1])) \
                        / rec["fs"] * 1000.0
        if self.return_cfg.rr_interval:
            out["rr_interval"] = float(rr_ms)
        if self.return_cfg.hr:
            out["hr"] = 60000.0 / rr_ms if rr_ms > 0 else 0.0

        if self.return_cfg.intervals:
            if "intervals" in rec:
                ivl = np.asarray(rec["intervals"][beat_idx], dtype=np.float32)
                out["intervals"]      = torch.from_numpy(ivl)
                out["interval_names"] = rec.get("interval_names", DEFAULT_INTERVAL_NAMES)
            else:
                n = len(DEFAULT_INTERVAL_NAMES)
                out["intervals"]      = torch.full((n,), float("nan"))
                out["interval_names"] = DEFAULT_INTERVAL_NAMES

        if self.return_cfg.meta:
            out["meta"] = {
                "cohort":     r.get("cohort", ""),
                "record_id":  r.get("record_id", ""),
                "subject_id": r.get("subject_id", ""),
                "beat_idx":   beat_idx,
            }

        if self.return_cfg.demo:
            out["demo"] = rec.get("demo", {})

        for fname in self.return_cfg.extra_npy:
            key = fname[:-4] if fname.endswith(".npy") else fname
            store_key = f"extra:{key}"
            if store_key in rec:
                arr = np.asarray(rec[store_key][beat_idx], dtype=np.float32)
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                out[key] = torch.from_numpy(arr)

        return out

    @property
    def n_records(self) -> int:     return len(self.records)
    @property
    def cohorts(self) -> List[str]: return sorted({r.get("cohort", "") for r in self.records})
    @property
    def subjects(self) -> List[str]: return sorted({r.get("subject_id", "") for r in self.records})


class HiCardiSequenceDataset(Dataset):
    """연속 beat 시퀀스 Dataset (세그먼트 안에서 슬라이딩 윈도우)."""

    def __init__(
        self,
        records:       List[Dict[str, str]],
        seq_len:       int                   = 16,
        stride:        int                   = 1,
        return_cfg:    ReturnConfig          = None,
        normalize:     str                   = "z",
        label_names:   Optional[List[str]]   = None,
        n_classes:     int                   = 0,
        cohort_filter: Optional[List[str]]   = None,
        min_beats:     int                   = 1,
    ):
        self.seq_len    = int(seq_len)
        self.stride     = int(stride)
        self.return_cfg = return_cfg or ReturnConfig()
        self.normalize  = normalize
        self.label_names = label_names or DEFAULT_BEAT_CLASSES
        self.n_classes   = n_classes or len(self.label_names)

        if cohort_filter is not None:
            allowed = set(cohort_filter)
            records = [r for r in records if r.get("cohort", "") in allowed]
        self.records = [r for r in records if int(r.get("num_beats", 0)) >= min_beats]
        self.store   = MemmapRecordStore(extra_npy=self.return_cfg.extra_npy)

        self.index: List[tuple] = []
        for ridx, r in enumerate(self.records):
            rec = self.store.get(r["cache_path"])
            for sid in range(int(rec["seg_start"].shape[0])):
                seg_len = int(rec["seg_len"][sid])
                s0      = int(rec["seg_start"][sid])
                if seg_len < self.seq_len:
                    continue
                for t in range(0, seg_len - self.seq_len + 1, self.stride):
                    self.index.append((ridx, s0 + t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ridx, start = self.index[idx]
        r   = self.records[ridx]
        rec = self.store.get(r["cache_path"])
        sl  = slice(start, start + self.seq_len)
        out: Dict[str, Any] = {}

        if self.return_cfg.waveform:
            seq = np.asarray(rec["X"][sl], dtype=np.float32)
            out["waveforms"] = torch.from_numpy(_normalize(seq, self.normalize))

        if self.return_cfg.labels:
            if "labels" in rec:
                lbl = np.asarray(rec["labels"][sl], dtype=np.float32)
                if lbl.shape[1] != self.n_classes:
                    cols = [HICARDI_LABEL_COL_MAP.get(n, i)
                            for i, n in enumerate(self.label_names)]
                    if all(c < lbl.shape[1] for c in cols):
                        lbl = lbl[:, cols]
                    else:
                        lbl = lbl[:, :self.n_classes]
            else:
                # sentinel -1: trainer가 해당 샘플을 필터링할 수 있도록 표시
                lbl = np.full((self.seq_len, self.n_classes), -1.0, dtype=np.float32)
            out["labels"] = torch.from_numpy(lbl)

        rr_seq: Optional[np.ndarray] = None
        if self.return_cfg.rr_interval or self.return_cfg.hr:
            ridx_arr = np.asarray(rec["r_idx"][sl], dtype=np.float32)
            rr_seq   = np.zeros(self.seq_len, dtype=np.float32)
            rr_seq[1:] = np.diff(ridx_arr) / rec["fs"] * 1000.0
        if self.return_cfg.rr_interval:
            out["rr_interval"] = torch.from_numpy(rr_seq)
        if self.return_cfg.hr:
            hr_seq = np.where(rr_seq > 0, 60000.0 / np.maximum(rr_seq, 1.0), 0.0)
            out["hr"] = torch.from_numpy(hr_seq.astype(np.float32))

        if self.return_cfg.intervals:
            if "intervals" in rec:
                ivl = np.asarray(rec["intervals"][sl], dtype=np.float32)
                out["intervals"]      = torch.from_numpy(ivl)
                out["interval_names"] = rec.get("interval_names", DEFAULT_INTERVAL_NAMES)
            else:
                out["intervals"]      = torch.full((self.seq_len, len(DEFAULT_INTERVAL_NAMES)), float("nan"))
                out["interval_names"] = DEFAULT_INTERVAL_NAMES

        if self.return_cfg.meta:
            out["meta"] = {
                "cohort":     r.get("cohort", ""),
                "record_id":  r.get("record_id", ""),
                "subject_id": r.get("subject_id", ""),
                "start_beat": start,
            }

        if self.return_cfg.demo:
            out["demo"] = rec.get("demo", {})

        for fname in self.return_cfg.extra_npy:
            key = fname[:-4] if fname.endswith(".npy") else fname
            store_key = f"extra:{key}"
            if store_key in rec:
                arr = np.asarray(rec[store_key][sl], dtype=np.float32)
                out[key] = torch.from_numpy(arr)

        return out

    @property
    def n_records(self) -> int: return len(self.records)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. IndexSubset — flat 인덱스 기반 Subset 래퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IndexSubset(Dataset):
    """HiCardiDataset의 flat 인덱스 필터 래퍼 (의미 중립적 명명)."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int: return len(self.indices)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[self.indices[idx]]

    @property
    def n_records(self) -> int:  return getattr(self.dataset, "n_records", 0)
    @property
    def cohorts(self):           return getattr(self.dataset, "cohorts", [])
    @property
    def subjects(self):          return getattr(self.dataset, "subjects", [])


# 하위호환 alias
BalancedSubset = IndexSubset


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. WaveformFetcher — 재사용 가능한 파형 일괄 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class WaveformFetcher:
    """flat 인덱스 → 파형 numpy 배열. centroid 계산용 max_samples 서브샘플링 지원."""

    @staticmethod
    def fetch(
        dataset:      HiCardiDataset,
        flat_indices: Sequence[int],
        max_samples:  int = 50_000,
        seed:         int = 42,
        label:        str = "파형 로딩",
    ) -> np.ndarray:
        flat_indices = list(flat_indices)
        if not flat_indices:
            return np.zeros((0,), dtype=np.float32)
        if len(flat_indices) > max_samples:
            rng = np.random.default_rng(seed)
            flat_indices = rng.choice(flat_indices, max_samples, replace=False).tolist()
        store = MemmapRecordStore()
        waves: List[np.ndarray] = []
        n = len(flat_indices)
        step = max(1, n // 20)
        for i, idx in enumerate(flat_indices):
            ridx, beat = dataset.index[idx]
            rec = store.get(dataset.records[ridx]["cache_path"])
            waves.append(np.asarray(rec["X"][beat], dtype=np.float32))
            if (i + 1) % step == 0 or (i + 1) == n:
                print(f"\r  {label}: {i+1:,}/{n:,}", end="", flush=True)
        print()
        return np.stack(waves)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. BeatLabelMasker — 정상/비정상 mask 계산 (단일 책임)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BeatLabelMasker:
    """레이블에서 BeatMask 계산.

    정상 정의: 병적 클래스 열이 모두 0인 beat.
    missing_as_normal=False(기본): 라벨 파일 없는 레코드는 normal/abnormal 어느 쪽에도
    포함하지 않아 파이프라인에서 자동 제외됨.
    missing_as_normal=True: 해당 레코드의 beat를 전부 정상으로 처리(cohort가 정상군임이
    보장된 경우에만 명시적으로 활성화).
    """

    def __init__(self, exclude_normal_col: bool = True, missing_as_normal: bool = False):
        self.exclude_normal_col = exclude_normal_col
        self.missing_as_normal  = missing_as_normal

    def compute(self, dataset: HiCardiDataset) -> BeatMask:
        store = MemmapRecordStore()
        normal_idx:   List[int] = []
        abnormal_idx: List[int] = []
        flat = 0
        n_records = len(dataset.records)
        step = max(1, n_records // 20)

        for ri, r in enumerate(dataset.records):
            n   = int(r.get("num_beats", 0))
            rec = store.get(r["cache_path"])
            if (ri + 1) % step == 0 or (ri + 1) == n_records:
                print(f"\r  레이블 분류: {ri+1}/{n_records} 레코드  ({flat:,} beats)", end="", flush=True)

            if "labels" not in rec:
                if self.missing_as_normal:
                    normal_idx.extend(range(flat, flat + n))
                # missing_as_normal=False: 파이프라인에서 제외 (flat 카운터는 진행)
            else:
                Y = np.asarray(rec["labels"], dtype=np.float32)
                n = min(n, len(Y))
                if Y.ndim == 1:
                    Y = Y.reshape(-1, 1)

                n_col = Y.shape[1]
                if self.exclude_normal_col and n_col > 1:
                    ab_cols = [c for c in _ABNORMAL_COLS if c < n_col]
                    check   = Y[:n, ab_cols] if ab_cols else Y[:n]
                else:
                    check = Y[:n]

                is_normal = check.sum(axis=1) == 0
                for b in range(n):
                    (normal_idx if is_normal[b] else abnormal_idx).append(flat + b)
            flat += n

        print()
        return BeatMask(normal=normal_idx, abnormal=abnormal_idx)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Strategy Pattern — Undersampling (Stage 1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IUndersampler(ABC):
    """정상 beat 풀에서 유지할 n_keep개의 flat 인덱스 선택."""

    @abstractmethod
    def select(
        self,
        dataset:        HiCardiDataset,
        normal_indices: List[int],
        n_keep:         int,
        cfg:            UndersamplingConfig,
    ) -> List[int]:
        ...


class RandomUndersampler(IUndersampler):
    def select(self, dataset, normal_indices, n_keep, cfg):
        rng = np.random.default_rng(cfg.seed)
        return rng.choice(normal_indices, n_keep, replace=False).tolist()


class _CentroidBasedUndersampler(IUndersampler):
    _far: bool = False

    _CHUNK = 10_000  # 청크당 ~20 MB (10K × 501 × float32)

    def select(self, dataset, normal_indices, n_keep, cfg):
        n = len(normal_indices)
        print(f"  센트로이드 계산 (서브샘플 최대 {cfg.centroid_max_samples:,}개 / 전체 {n:,}개)")
        X_sub = WaveformFetcher.fetch(
            dataset, normal_indices, cfg.centroid_max_samples, cfg.seed,
            label="센트로이드 서브샘플")
        c = X_sub.mean(axis=0).astype(np.float32)
        del X_sub

        # 전체 정상 샘플에 대한 거리를 청크 단위로 계산 (OOM 방지)
        distances = np.empty(n, dtype=np.float32)
        store = MemmapRecordStore()
        for start in range(0, n, self._CHUNK):
            end = min(start + self._CHUNK, n)
            waves = [
                np.asarray(
                    store.get(dataset.records[dataset.index[idx][0]]["cache_path"])
                    ["X"][dataset.index[idx][1]],
                    dtype=np.float32,
                )
                for idx in normal_indices[start:end]
            ]
            distances[start:end] = compute_distances(np.stack(waves), c, cfg.distance_metric)
            print(f"\r  거리 계산: {end:,}/{n:,}", end="", flush=True)
        print()

        order = np.argsort(distances)
        if self._far:
            order = order[::-1]
        return [normal_indices[i] for i in order[:n_keep]]


class CentroidNearUndersampler(_CentroidBasedUndersampler):
    _far = False


class CentroidFarUndersampler(_CentroidBasedUndersampler):
    _far = True


class DistThresholdUndersampler(IUndersampler):
    def select(self, dataset, normal_indices, n_keep, cfg):
        X_all = WaveformFetcher.fetch(
            dataset, normal_indices, max_samples=len(normal_indices), seed=cfg.seed)
        c = X_all[: cfg.centroid_max_samples].mean(axis=0)
        d = compute_distances(X_all, c, cfg.distance_metric)
        thr = (d >= cfg.dist_threshold) if cfg.keep_above_threshold else (d < cfg.dist_threshold)
        cands = [normal_indices[i] for i in range(len(normal_indices)) if thr[i]]
        return cands[:n_keep] if len(cands) > n_keep else cands


_UNDERSAMPLER_REGISTRY: Dict[SamplingStrategy, type] = {
    SamplingStrategy.RANDOM:         RandomUndersampler,
    SamplingStrategy.CENTROID_NEAR:  CentroidNearUndersampler,
    SamplingStrategy.CENTROID_FAR:   CentroidFarUndersampler,
    SamplingStrategy.DIST_THRESHOLD: DistThresholdUndersampler,
}


def make_undersampler(strategy: SamplingStrategy) -> Optional[IUndersampler]:
    if strategy == SamplingStrategy.NONE:
        return None
    cls = _UNDERSAMPLER_REGISTRY.get(strategy)
    if cls is None:
        raise ValueError(f"알 수 없는 SamplingStrategy: {strategy}")
    return cls()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Strategy Pattern — Prototyping (Stage 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IPrototyper(ABC):
    """후보 풀에서 n_keep개 선택. EfficiencyStage가 클래스별로 각각 호출."""

    @abstractmethod
    def select(
        self,
        dataset:           HiCardiDataset,
        candidate_indices: List[int],
        n_keep:            int,
        distance_metric:   DistanceMetric,
        seed:              int,
    ) -> List[int]:
        ...


class RandomPrototyper(IPrototyper):
    def select(self, dataset, candidate_indices, n_keep, distance_metric, seed):
        if n_keep >= len(candidate_indices):
            return list(candidate_indices)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(candidate_indices), n_keep, replace=False)
        return [candidate_indices[i] for i in chosen]


class _CentroidBasedPrototyper(IPrototyper):
    _far: bool = False

    def select(self, dataset, candidate_indices, n_keep, distance_metric, seed):
        if n_keep >= len(candidate_indices):
            return list(candidate_indices)
        X = WaveformFetcher.fetch(
            dataset, candidate_indices, max_samples=len(candidate_indices), seed=seed)
        c = X[: min(50_000, len(X))].mean(axis=0)
        d = compute_distances(X, c, distance_metric)
        order = np.argsort(d)
        if self._far:
            order = order[::-1]
        return [candidate_indices[i] for i in order[:n_keep]]


class CentroidPrototyper(_CentroidBasedPrototyper):
    _far = False


class HardMiningPrototyper(_CentroidBasedPrototyper):
    _far = True


_PROTOTYPER_REGISTRY: Dict[PrototypingStrategy, type] = {
    PrototypingStrategy.RANDOM:      RandomPrototyper,
    PrototypingStrategy.CENTROID:    CentroidPrototyper,
    PrototypingStrategy.HARD_MINING: HardMiningPrototyper,
}


def make_prototyper(strategy: PrototypingStrategy) -> IPrototyper:
    cls = _PROTOTYPER_REGISTRY.get(strategy)
    if cls is None:
        raise ValueError(f"알 수 없는 PrototypingStrategy: {strategy}")
    return cls()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. Pipeline Stages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UndersamplingStage:
    """Stage 1: 정상/비정상 비율을 cfg.normal_ratio에 맞춤."""

    def __init__(self, cfg: UndersamplingConfig, masker: BeatLabelMasker):
        self.cfg    = cfg
        self.masker = masker

    def apply(
        self,
        dataset:      HiCardiDataset,
        initial_mask: Optional[BeatMask] = None,
    ) -> StageResult:
        mask = initial_mask if initial_mask is not None else self.masker.compute(dataset)
        sampler = make_undersampler(self.cfg.strategy)

        if sampler is None or self.cfg.normal_ratio <= 0 or not mask.normal:
            return self._passthrough(mask)

        n_keep = min(len(mask.normal),
                     max(1, int(len(mask.abnormal) * self.cfg.normal_ratio)))

        if n_keep >= len(mask.normal):
            return self._passthrough(mask)

        kept_normal = sorted(set(sampler.select(dataset, mask.normal, n_keep, self.cfg)))
        new_mask = BeatMask(normal=kept_normal, abnormal=list(mask.abnormal))
        return StageResult(
            indices=sorted(new_mask.all),
            mask=new_mask,
            applied=True,
            name="undersampling",
        )

    @staticmethod
    def _passthrough(mask: BeatMask) -> StageResult:
        return StageResult(
            indices=sorted(mask.all),
            mask=BeatMask(normal=list(mask.normal), abnormal=list(mask.abnormal)),
            applied=False,
            name="undersampling",
        )


class EfficiencyStage:
    """Stage 2: stratified subsampling — normal/abnormal 풀에 prototyping을 각각 적용
    하여 비율을 보존하면서 축소.

    이전 설계의 핵심 결함(centroid/hard_mining이 Stage 1 균형을 깨뜨림)을 해결.
    """

    def __init__(self, cfg: EfficiencyConfig, distance_metric: DistanceMetric):
        self.cfg = cfg
        self.distance_metric = distance_metric
        self.prototyper = make_prototyper(cfg.prototyping)

    def apply(
        self,
        dataset:    HiCardiDataset,
        prev_stage: StageResult,
    ) -> StageResult:
        n_current = len(prev_stage.indices)
        if self.cfg.variation_factor >= 1.0 or n_current < self.cfg.apply_threshold:
            return self._passthrough(prev_stage)

        # 클래스별 stratified subsampling
        normal_pool   = list(prev_stage.mask.normal)
        abnormal_pool = list(prev_stage.mask.abnormal)
        f = self.cfg.variation_factor

        n_keep_n = max(1, int(len(normal_pool)   * f)) if normal_pool   else 0
        n_keep_a = max(1, int(len(abnormal_pool) * f)) if abnormal_pool else 0

        kept_n = self.prototyper.select(
            dataset, normal_pool,   n_keep_n, self.distance_metric, self.cfg.seed,
        ) if normal_pool else []
        kept_a = self.prototyper.select(
            dataset, abnormal_pool, n_keep_a, self.distance_metric, self.cfg.seed + 1,
        ) if abnormal_pool else []

        new_mask = BeatMask(normal=sorted(kept_n), abnormal=sorted(kept_a))
        return StageResult(
            indices=sorted(new_mask.all),
            mask=new_mask,
            applied=True,
            name="efficiency",
        )

    @staticmethod
    def _passthrough(prev: StageResult) -> StageResult:
        return StageResult(
            indices=list(prev.indices),
            mask=BeatMask(normal=list(prev.mask.normal),
                          abnormal=list(prev.mask.abnormal)),
            applied=False,
            name="efficiency",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. DataPipeline — Stage 오케스트레이터 (단일 진입점)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DataPipeline:
    """raw_dataset → undersampling → efficiency → active_dataset

    핵심: setup과 resample이 동일하게 이 한 메서드(run)만 호출함으로써
          stage 일부가 우회되는 버그를 구조적으로 차단.
    """

    def __init__(
        self,
        masker:    BeatLabelMasker,
        und_stage: Optional[UndersamplingStage] = None,
        eff_stage: Optional[EfficiencyStage]    = None,
    ):
        self.masker    = masker
        self.und_stage = und_stage
        self.eff_stage = eff_stage

    def run(self, raw_dataset: HiCardiDataset) -> PipelineResult:
        initial_mask = self.masker.compute(raw_dataset)

        stage1 = (self.und_stage.apply(raw_dataset, initial_mask)
                  if self.und_stage is not None
                  else UndersamplingStage._passthrough(initial_mask))

        stage2 = (self.eff_stage.apply(raw_dataset, stage1)
                  if self.eff_stage is not None
                  else EfficiencyStage._passthrough(stage1))

        active = (IndexSubset(raw_dataset, stage2.indices)
                  if (stage1.applied or stage2.applied)
                  else raw_dataset)

        return PipelineResult(
            raw_dataset=raw_dataset,
            initial_mask=initial_mask,
            stage1_result=stage1,
            stage2_result=stage2,
            active_dataset=active,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. MixupCollate — beat & sequence 양쪽 지원
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MixupCollate:
    """배치 단위 Mixup. beat 모드("waveform")와 sequence 모드("waveforms") 양쪽 지원."""

    _WAVE_KEYS = ("waveform", "waveforms")

    def __init__(self, base_collate, cfg: MixupConfig, seed: int = 42):
        self.base_collate = base_collate
        self.cfg          = cfg
        self._seed        = seed
        self._rng         = np.random.default_rng(seed)
        self._init_worker = False

    def _ensure_worker_rng(self) -> None:
        """worker별 독립 시드 — 동일 lambda 시퀀스 중복 방지."""
        if self._init_worker:
            return
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        self._rng = np.random.default_rng(self._seed + wid * 1000003)
        self._init_worker = True

    def __call__(self, batch):
        out = self.base_collate(batch)
        if not self.cfg.is_active:
            return out

        self._ensure_worker_rng()

        wave_key = next((k for k in self._WAVE_KEYS if k in out), None)
        if wave_key is None:
            return out
        if self._rng.random() >= self.cfg.apply_prob:
            return out

        lam  = float(self._rng.beta(self.cfg.alpha, self.cfg.alpha))
        B    = out[wave_key].shape[0]
        perm = torch.randperm(B)

        out[wave_key] = lam * out[wave_key] + (1.0 - lam) * out[wave_key][perm]
        if "labels" in out:
            out["labels"] = lam * out["labels"] + (1.0 - lam) * out["labels"][perm]
        out["mixup_lambda"] = torch.tensor(lam, dtype=torch.float32)
        return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 14. LoaderFactory — DataLoader 생성 책임 분리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LoaderFactory:
    """DataLoader 생성. train에만 Mixup 적용."""

    def __init__(self, cfg: DataLoaderConfig, mixup_cfg: MixupConfig, mixup_seed: int = 42):
        self.cfg        = cfg
        self.mixup_cfg  = mixup_cfg
        self.mixup_seed = mixup_seed

    def _train_collate(self):
        if self.mixup_cfg.is_active:
            return MixupCollate(hicardi_collate_fn, self.mixup_cfg, self.mixup_seed)
        return hicardi_collate_fn

    def _make(self, dataset: Dataset, shuffle: bool, drop_last: bool, collate) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size  = self.cfg.batch_size,
            shuffle     = shuffle,
            num_workers = self.cfg.num_workers,
            pin_memory  = self.cfg.pin_memory,
            drop_last   = drop_last,
            persistent_workers = (self.cfg.num_workers > 0),
            collate_fn  = collate,
        )

    def train_loader(self, ds: Dataset) -> DataLoader:
        return self._make(ds, shuffle=True, drop_last=self.cfg.drop_last,
                          collate=self._train_collate())

    def eval_loader(self, ds: Dataset) -> DataLoader:
        return self._make(ds, shuffle=False, drop_last=False,
                          collate=hicardi_collate_fn)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 15. RecordCentroidAnalyzer — 분석 도구 (기존 유지)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RecordCentroidAnalyzer:
    """레코드/코호트별 ECG 파형 센트로이드 및 거리 통계 분석."""

    def __init__(self, dataset: HiCardiDataset):
        self.dataset = dataset
        self._store  = MemmapRecordStore()

    def record_centroids(self) -> Dict[str, np.ndarray]:
        out = {}
        for r in self.dataset.records:
            rec = self._store.get(r["cache_path"])
            out[r["record_id"]] = np.asarray(rec["X"], dtype=np.float32).mean(axis=0)
        return out

    def cohort_centroids(self) -> Dict[str, np.ndarray]:
        buf: Dict[str, List[np.ndarray]] = {}
        for r in self.dataset.records:
            cohort = r.get("cohort", "unknown")
            rec    = self._store.get(r["cache_path"])
            buf.setdefault(cohort, []).append(np.asarray(rec["X"], dtype=np.float32))
        return {c: np.concatenate(arrs).mean(axis=0) for c, arrs in buf.items()}

    def class_centroid(self, label_col: int) -> Optional[np.ndarray]:
        beats: List[np.ndarray] = []
        for r in self.dataset.records:
            rec = self._store.get(r["cache_path"])
            if "labels" not in rec:
                continue
            X = np.asarray(rec["X"],      dtype=np.float32)
            Y = np.asarray(rec["labels"], dtype=np.float32)
            if Y.ndim > 1 and label_col < Y.shape[1]:
                beats.append(X[Y[:, label_col] > 0])
        if not beats:
            return None
        all_b = np.concatenate(beats)
        return all_b.mean(axis=0) if len(all_b) > 0 else None

    def distance_stats(
        self,
        level:  str            = "record",
        metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
    ) -> Dict[str, Dict[str, float]]:
        centroids = self.record_centroids() if level == "record" else self.cohort_centroids()
        key_fn = ((lambda r: r["record_id"]) if level == "record"
                  else (lambda r: r.get("cohort", "unknown")))

        stats: Dict[str, Dict[str, Any]] = {}
        for r in self.dataset.records:
            key = key_fn(r)
            if key not in centroids:
                continue
            rec = self._store.get(r["cache_path"])
            X   = np.asarray(rec["X"], dtype=np.float32)
            d   = compute_distances(X, centroids[key], metric)
            entry = stats.setdefault(key, {"n_beats": 0, "_all": []})
            entry["n_beats"] += len(d)
            entry["_all"].extend(d.tolist())

        out: Dict[str, Dict[str, float]] = {}
        for key, e in stats.items():
            arr = np.array(e["_all"])
            out[key] = {
                "n_beats": int(e["n_beats"]),
                "mean":    float(arr.mean()),
                "std":     float(arr.std()),
                "p25":     float(np.percentile(arr, 25)),
                "p50":     float(np.percentile(arr, 50)),
                "p75":     float(np.percentile(arr, 75)),
                "min":     float(arr.min()),
                "max":     float(arr.max()),
            }
        return out

    def print_stats(self, level="record", metric=DistanceMetric.EUCLIDEAN) -> None:
        print(f"\n센트로이드 거리 통계 ({level}별, {metric.value})")
        print("=" * 72)
        for key, s in sorted(self.distance_stats(level, metric).items()):
            print(f"  {key:<32s}  n={s['n_beats']:6d}  "
                  f"mean={s['mean']:.4f}  std={s['std']:.4f}  p50={s['p50']:.4f}")

    def export_centroids(self, save_dir: str, level: str = "record") -> Dict[str, str]:
        os.makedirs(save_dir, exist_ok=True)
        centroids = self.record_centroids() if level == "record" else self.cohort_centroids()
        paths: Dict[str, str] = {}
        for name, c in centroids.items():
            fname = name.replace("/", "_").replace(" ", "_") + "_centroid.npy"
            fpath = os.path.join(save_dir, fname)
            np.save(fpath, c)
            paths[name] = fpath
        return paths


