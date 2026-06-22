# -*- coding: utf-8 -*-
"""
Holter Beat CPC Pretraining Pipeline (FAST + Leakage-safe + Segmented sequences)

Implemented requirements
1) Beat chunk extraction around R-peak (center-based): pre/post samples (default 360/360 => 721).
2) Fast cache: per-record .npy arrays + memmap loading (avoid NPZ decompression overhead).
3) Sequence segmentation: split beat stream into segments by RR gap threshold (seconds).
4) Subject leakage prevention: train/val/test split by subject_id (group split).
5) Negative sampling:
   - In-batch negatives (default, fast)
   - Optional cross-record negative beats (from other records, memmap)
6) Two dataset modes:
   - InfoNCE CPC (context K, predict P future beats)
   - CPC v2 masked prediction (mask beats in a sequence, predict masked beats)

Assumptions
- Each record is MATLAB -v7.3 HDF5 .mat with keys:
  dECG, fs, data_lost, LeadOff, and one of Rpk_flag / Rpk_label / RpkFlag / RpkLabel

Directory layout example (ROOT_DIR):
F:/
  01.Holter(Local Clinic)/
  02.Holter(Kosin)/
  ...
  07.Holter(Yonsei_Baek_2023)/

Cache layout (CACHE_DIR):
CACHE_DIR/
  index.csv
  cohorts/<cohort>/<record_id>/
     X.npy            float16 or float32, shape (B, win_len)
     r_idx.npy        int32, shape (B,)
     seg_id.npy       int32, shape (B,)
     seg_start.npy    int32, shape (S,)  (segment starts in beat-index space)
     seg_len.npy      int32, shape (S,)  (segment lengths)
     fs.npy           float32, shape (1,)
     meta.txt         small text (optional)

Usage outline
1) Build cache once:
   python this_file.py --build_cache --root "F:\\" --cache "./beat_cache"
2) Create split:
   python this_file.py --make_split --cache "./beat_cache" --split "./split.json"
3) Train:
   - Use datasets provided below; DataLoader with num_workers/pin_memory

Notes
- subject_id extraction is heuristic by default; customize extract_subject_id() as needed.
- RR segmentation uses r_idx diffs; also respects quality mask (data_lost, lead_off) by removing invalid r-peaks.
"""

from __future__ import annotations

import os
import csv
import json
import glob
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import h5py

import torch
from torch.utils.data import Dataset, DataLoader
ROOT_DIR = "Database/hicardi"
CACHE_DIR = "Database/hicardi_beat_cache"
SPLIT_JSON_PATH = "Database/hicardi_beat_cache/split.json"

# =============================================================================
# 실행 모드 (여기서만 변경)
# =============================================================================
#   "build"     → 캐시 생성 + split 생성 (기존 동작)
#   "diagnose"  → .mat 파일 내부 구조 탐색 (Y.npy 누락 원인 확인용)
#   "verify"    → 라벨 추출 메모리 검증 (디스크 저장 없음)
#   "plot"      → ECG + R-peak 시각화 이미지 저장
RUN_MODE = "build"

# "diagnose" / "verify" / "plot" 모드에서 사용할 .mat 파일 경로
DIAGNOSE_MAT_PATH = "Database/hicardi/01.Holter(Local Clinic), 124 records)/DN124_7972_20210907.mat"

# "plot" 모드 설정
PLOT_SAVE_DIR  = "./test_samples/"
PLOT_NUM_FILES = 5
PLOT_DURATION_SEC = 15.0
# =============================================================================

# 실행할 파이프라인 단계 (True/False)
RUN_BUILD_CACHE = True
RUN_MAKE_SPLIT = True
RUN_DEMO_LOADER = False

# 캐시(Cache) 생성 파라미터
PRE_SAMPLES = 250
POST_SAMPLES = 250
RR_GAP_LIMIT_SEC = 2.0
MIN_SEG_BEATS = 4
DTYPE_STORE = "float32"  # "float16" or "float32"
OVERWRITE_CACHE = False

# 분할(Split) 생성 파라미터
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# =============================================================================
# 0) Utilities
# =============================================================================
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _as_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and 1 in x.shape:
        return x.reshape(-1)
    return x.reshape(-1)

def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# =============================================================================
# 1) Minimal Holter loader (v7.3 HDF5)
# =============================================================================
@dataclass
class HolterRecord:
    fs: float
    ecg_mv: np.ndarray      # (n,) float32
    rpeak: np.ndarray       # (n,) int8
    data_lost: np.ndarray   # (n,) int8
    lead_off: np.ndarray    # (n,) int8
    final_flag: Optional[np.ndarray] = None  # (n, 27) int8; None if not present

    @property
    def n(self) -> int:
        return int(self.ecg_mv.shape[0])

def load_holter_mat_v73_minimal(mat_path: str, ecg_mv_dtype=np.float32,
                                verbose: bool = False) -> HolterRecord:
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(mat_path)

    with h5py.File(mat_path, "r") as f:

        def get_obj(name: str):
            if name in f:
                return f[name]
            for k in f.keys():
                if k.lower() == name.lower():
                    return f[k]
            raise KeyError(f"'{name}' not found. Available keys: {list(f.keys())}")

        dECG = np.array(get_obj("dECG"))
        fs = float(np.array(get_obj("fs")).reshape(-1)[0])
        data_lost = np.array(get_obj("data_lost"))
        lead_off = np.array(get_obj("LeadOff"))

        rpk_flag = None
        for cand in ["Rpk_flag", "Rpk_label", "RpkFlag", "RpkLabel"]:
            try:
                rpk_flag = np.array(get_obj(cand))
                break
            except KeyError:
                continue
        if rpk_flag is None:
            raise KeyError("R-peak variable not found. Tried: Rpk_flag / Rpk_label / RpkFlag / RpkLabel")

        # Load supervised beat labels (27 classes). HDF5 may store as (27, n) → transpose.
        # ── 실패 구간 1: Key 이름 불일치 ────────────────────────────────────────
        _final_flag_raw = None
        try:
            ff = np.array(get_obj("final_flag"))
            if verbose:
                print(f"  [diag] ✓ final_flag 발견  shape={ff.shape}  dtype={ff.dtype}")

            # ── 실패 구간 2: 차원(shape) 불일치 ───────────────────────────────
            if ff.ndim == 2:
                if ff.shape[0] == 27 and ff.shape[1] != 27:
                    ff = ff.T   # (27, n) → (n, 27)
                    if verbose:
                        print(f"  [diag] 전치(Transpose) 적용 → {ff.shape}")
                if ff.shape[1] == 27:
                    ff = np.where(ff > 0, 1, 0)
                    _final_flag_raw = ff.astype(np.int8, copy=False)
                    if verbose:
                        print(f"  [diag] ✓ 이진화 완료 → _final_flag_raw {_final_flag_raw.shape}")
                else:
                    if verbose:
                        print(f"  [diag] ✗ 실패 구간 2: shape[1]={ff.shape[1]} ≠ 27 → 라벨 무시")
            else:
                if verbose:
                    print(f"  [diag] ✗ 실패 구간 2: ndim={ff.ndim} ≠ 2 (shape={ff.shape}) → 라벨 무시")
        except KeyError as e:
            if verbose:
                print(f"  [diag] ✗ 실패 구간 1: KeyError — {e}")

    # ecg_adc = _as_1d(dECG).astype(np.int32, copy=False)
    # ecg_mv = (ecg_adc - 8192) / 1000.0
    # ecg_mv = ecg_mv.astype(ecg_mv_dtype, copy=False)

    # rpeak = _as_1d(rpk_flag).astype(np.int8, copy=False)
    # data_lost_1d = _as_1d(data_lost).astype(np.int8, copy=False)
    # lead_off_1d = _as_1d(lead_off).astype(np.int8, copy=False)

    ecg_adc = np.nan_to_num(_as_1d(dECG), nan=8192.0, posinf=8192.0, neginf=8192.0).astype(np.int32, copy=False)
    ecg_mv = (ecg_adc - 8192) / 1000.0
    ecg_mv = ecg_mv.astype(ecg_mv_dtype, copy=False)

    # 2. 라벨 및 품질 마스크 결측치 방어
    rpeak = np.nan_to_num(_as_1d(rpk_flag), nan=0.0).astype(np.int8, copy=False)
    
    # 품질 불량 지표(data_lost, lead_off)에 있는 NaN은 '불량(1)'으로 간주
    data_lost_1d = np.nan_to_num(_as_1d(data_lost), nan=1.0).astype(np.int8, copy=False)
    lead_off_1d = np.nan_to_num(_as_1d(lead_off), nan=1.0).astype(np.int8, copy=False)
    n = ecg_mv.shape[0]
    if rpeak.shape[0] != n:
        raise ValueError(f"Length mismatch: ecg n={n}, rpeak n={rpeak.shape[0]}")
    if data_lost_1d.shape[0] != n:
        raise ValueError(f"Length mismatch: ecg n={n}, data_lost n={data_lost_1d.shape[0]}")
    if lead_off_1d.shape[0] != n:
        raise ValueError(f"Length mismatch: ecg n={n}, lead_off n={lead_off_1d.shape[0]}")

    # ── 실패 구간 3: 신호 길이 ≠ 라벨 길이 ─────────────────────────────────────
    if _final_flag_raw is not None and _final_flag_raw.shape[0] != n:
        import warnings
        msg = (f"final_flag length {_final_flag_raw.shape[0]} != ECG length {n}; ignoring labels")
        warnings.warn(msg, stacklevel=2)
        if verbose:
            print(f"  [diag] ✗ 실패 구간 3: {msg}")
        _final_flag_raw = None
    elif _final_flag_raw is not None and verbose:
        print(f"  [diag] ✓ 실패 구간 3 통과: 길이 일치 ({n}샘플)")

    return HolterRecord(
        fs=fs,
        ecg_mv=ecg_mv,
        rpeak=rpeak,
        data_lost=data_lost_1d,
        lead_off=lead_off_1d,
        final_flag=_final_flag_raw,
    )

def build_valid_mask_from_quality(rec: HolterRecord, exclude_data_lost=True, exclude_lead_off=True) -> np.ndarray:
    valid = np.ones(rec.n, dtype=bool)
    if exclude_data_lost:
        valid &= (rec.data_lost == 0)
    if exclude_lead_off:
        valid &= (rec.lead_off == 0)
    return valid


# =============================================================================
# 2) Subject ID extraction (customize if needed)
# =============================================================================
def extract_subject_id_default(cohort: str, record_id: str) -> str:
    """
    Heuristic subject_id parser.
    You MUST customize if your record naming encodes subject differently.

    Default:
    - Take prefix before first '_' or '-' if present, else full record_id.
    - Include cohort to avoid accidental collisions across cohorts.
    """
    rid = record_id
    for sep in ["_", "-"]:
        if sep in rid:
            rid = rid.split(sep)[0]
            break
    return f"{cohort}::{rid}"


# =============================================================================
# 3) Beat extraction + segmentation
# =============================================================================
def extract_beats_from_record(
    rec: HolterRecord,
    valid_mask: Optional[np.ndarray],
    pre: int,
    post: int,
    min_valid_ratio: float,
    use_only_valid_rpeaks: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      X:           (B, win_len) float32
      r_idx:       (B,) int32
      valid_ratio: (B,) float32
      Y:           (B, 27) int8  — label at each R-peak; None if final_flag absent
    """
    win_len = pre + post + 1
    has_labels = rec.final_flag is not None

    rp = np.where(rec.rpeak > 0)[0]
    if rp.size == 0:
        return (np.zeros((0, win_len), np.float32),
                np.zeros((0,), np.int32),
                np.zeros((0,), np.float32),
                np.zeros((0, 27), np.int8) if has_labels else None)

    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if use_only_valid_rpeaks:
            rp = rp[vm[rp]]
    else:
        vm = None

    X_list: List[np.ndarray] = []
    ridx_list: List[int] = []
    vratio_list: List[float] = []
    Y_list: List[np.ndarray] = []

    for r in rp:
        s = int(r - pre)
        e = int(r + post + 1)  # exclusive
        if s < 0 or e > rec.n:
            continue
        x = rec.ecg_mv[s:e]
        if x.shape[0] != win_len:
            continue

        if vm is None:
            vratio = 1.0
        else:
            vratio = float(np.mean(vm[s:e]))
            if vratio < min_valid_ratio:
                continue

        X_list.append(x.astype(np.float32, copy=False))
        ridx_list.append(int(r))
        vratio_list.append(vratio)
        if has_labels:
            Y_list.append(rec.final_flag[int(r)])

    if len(X_list) == 0:
        return (np.zeros((0, win_len), np.float32),
                np.zeros((0,), np.int32),
                np.zeros((0,), np.float32),
                np.zeros((0, 27), np.int8) if has_labels else None)

    X = np.stack(X_list, axis=0)  # (B, win_len)
    r_idx = np.asarray(ridx_list, dtype=np.int32)
    valid_ratio = np.asarray(vratio_list, dtype=np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.int8, copy=False) if has_labels else None  # (B, 27)
    return X, r_idx, valid_ratio, Y

def segment_beats_by_rr_gap(
    r_idx: np.ndarray,
    fs: float,
    rr_gap_limit_sec: float,
    min_segment_beats: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment beats into contiguous segments where consecutive RR gap <= threshold.

    Returns:
      seg_id: (B,) int32
      seg_start: (S,) int32  start beat index per segment
      seg_len: (S,) int32    length per segment
    """
    B = int(r_idx.shape[0])
    if B == 0:
        return (np.zeros((0,), np.int32),
                np.zeros((0,), np.int32),
                np.zeros((0,), np.int32))

    gaps = np.diff(r_idx.astype(np.int64)) / float(fs)  # (B-1,)
    cut = np.zeros((B,), dtype=bool)
    cut[0] = True
    if B > 1:
        cut[1:] = gaps <= float(rr_gap_limit_sec)
        # cut[i]=True means it's contiguous from i-1 to i; we want starts where NOT contiguous
        starts = [0]
        for i in range(1, B):
            if not cut[i]:
                starts.append(i)
    else:
        starts = [0]

    starts = np.asarray(starts, dtype=np.int32)
    ends = np.append(starts[1:], np.array([B], dtype=np.int32))
    lens = (ends - starts).astype(np.int32)

    # filter segments by min length
    keep = lens >= int(min_segment_beats)
    starts_k = starts[keep]
    lens_k = lens[keep]

    seg_id = np.full((B,), -1, dtype=np.int32)
    sid = 0
    for s, ln in zip(starts_k.tolist(), lens_k.tolist()):
        seg_id[s:s+ln] = sid
        sid += 1

    return seg_id, starts_k, lens_k


# =============================================================================
# 4) Cache builder (FAST: .npy + memmap)
# =============================================================================
def discover_records(root_dir: str, exts: Tuple[str, ...] = (".mat",)) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for cohort in sorted(os.listdir(root_dir)):
        cohort_path = os.path.join(root_dir, cohort)
        if not os.path.isdir(cohort_path):
            continue
        for ext in exts:
            for p in glob.glob(os.path.join(cohort_path, f"*{ext}")):
                rid = os.path.splitext(os.path.basename(p))[0]
                out.append((cohort, rid, p))
    return out

def write_record_cache(
    out_dir: str,
    X: np.ndarray,
    r_idx: np.ndarray,
    seg_id: np.ndarray,
    seg_start: np.ndarray,
    seg_len: np.ndarray,
    fs: float,
    dtype_store: str = "float16",
    Y: Optional[np.ndarray] = None,
) -> None:
    safe_mkdir(out_dir)

    if dtype_store == "float16":
        X_store = X.astype(np.float16, copy=False)
    elif dtype_store == "float32":
        X_store = X.astype(np.float32, copy=False)
    else:
        raise ValueError("dtype_store must be 'float16' or 'float32'")

    np.save(os.path.join(out_dir, "X.npy"), X_store, allow_pickle=False)
    np.save(os.path.join(out_dir, "r_idx.npy"), r_idx.astype(np.int32, copy=False), allow_pickle=False)
    np.save(os.path.join(out_dir, "seg_id.npy"), seg_id.astype(np.int32, copy=False), allow_pickle=False)
    np.save(os.path.join(out_dir, "seg_start.npy"), seg_start.astype(np.int32, copy=False), allow_pickle=False)
    np.save(os.path.join(out_dir, "seg_len.npy"), seg_len.astype(np.int32, copy=False), allow_pickle=False)
    np.save(os.path.join(out_dir, "fs.npy"), np.asarray([fs], dtype=np.float32), allow_pickle=False)
    # Y.npy: (B, 27) int8 — only written when final_flag is present in the source .mat
    if Y is not None:
        np.save(os.path.join(out_dir, "Y.npy"), Y.astype(np.int8, copy=False), allow_pickle=False)

def build_cache(
    root_dir: str,
    cache_dir: str,
    pre: int = 360,
    post: int = 360,
    min_valid_ratio: float = 1.0,
    use_only_valid_rpeaks: bool = True,
    exclude_data_lost: bool = True,
    exclude_lead_off: bool = True,
    rr_gap_limit_sec: float = 2.0,
    min_segment_beats: int = 4,
    dtype_store: str = "float16",
    overwrite: bool = False,
) -> str:
    """
    Build cache and write index.csv.

    index.csv columns:
      cohort,record_id,subject_id,record_path,cache_path,num_beats,num_segments
    """
    safe_mkdir(cache_dir)
    index_path = os.path.join(cache_dir, "index.csv")

    records = discover_records(root_dir, exts=(".mat",))
    print(f"[Cache] discovered records: {len(records)}")

    rows: List[List[str]] = []
    for cohort, record_id, record_path in records:
        out_dir = os.path.join(cache_dir, "cohorts", cohort, record_id)

        # Skip if exists
        if (not overwrite) and os.path.isfile(os.path.join(out_dir, "X.npy")):
            # still index it
            try:
                X_shape = np.load(os.path.join(out_dir, "X.npy"), mmap_mode="r").shape
                seg_start = np.load(os.path.join(out_dir, "seg_start.npy"), mmap_mode="r")
                subject_id = extract_subject_id_default(cohort, record_id)
                rows.append([
                    cohort, record_id, subject_id, record_path, out_dir,
                    str(int(X_shape[0])), str(int(seg_start.shape[0]))
                ])
            except Exception:
                pass
            continue

        try:
            rec = load_holter_mat_v73_minimal(record_path)
            valid = build_valid_mask_from_quality(rec, exclude_data_lost, exclude_lead_off)
            X, r_idx, _vr, Y = extract_beats_from_record(
                rec, valid_mask=valid,
                pre=pre, post=post,
                min_valid_ratio=min_valid_ratio,
                use_only_valid_rpeaks=use_only_valid_rpeaks,
            )
            if X.shape[0] == 0:
                continue

            seg_id, seg_start, seg_len = segment_beats_by_rr_gap(
                r_idx=r_idx, fs=rec.fs,
                rr_gap_limit_sec=rr_gap_limit_sec,
                min_segment_beats=min_segment_beats,
            )
            # keep only beats that belong to kept segments (seg_id!=-1)
            keep = seg_id >= 0
            X = X[keep]
            r_idx = r_idx[keep]
            seg_id = seg_id[keep]
            if Y is not None:
                Y = Y[keep]

            # recompute seg_start/seg_len for compactness after filtering
            # Build mapping by scanning seg_id
            if X.shape[0] == 0:
                continue
            # seg_id are 0..S-1 in order already, but after filtering beats,
            # we recompute starts and lens in beat-index space.
            sid = seg_id.astype(np.int32)
            changes = np.where(np.diff(sid, prepend=sid[0]) != 0)[0]
            starts2 = changes.astype(np.int32)
            if starts2.size == 0 or starts2[0] != 0:
                starts2 = np.insert(starts2, 0, 0).astype(np.int32)
            # ends
            ends2 = np.append(starts2[1:], np.array([sid.shape[0]], dtype=np.int32))
            lens2 = (ends2 - starts2).astype(np.int32)

            write_record_cache(
                out_dir=out_dir, X=X, r_idx=r_idx,
                seg_id=sid, seg_start=starts2, seg_len=lens2,
                fs=rec.fs, dtype_store=dtype_store, Y=Y,
            )

            subject_id = extract_subject_id_default(cohort, record_id)
            rows.append([
                cohort, record_id, subject_id, record_path, out_dir,
                str(int(X.shape[0])), str(int(starts2.shape[0]))
            ])
            print(f"[Cache] {cohort}/{record_id}: beats={X.shape[0]} segs={starts2.shape[0]}")
        except Exception as e:
            print(f"[Cache][FAIL] {record_path} :: {type(e).__name__}: {e}")
            continue

    # write index.csv
    safe_mkdir(os.path.dirname(index_path) or ".")
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "record_id", "subject_id", "record_path", "cache_path", "num_beats", "num_segments"])
        for r in rows:
            w.writerow(r)

    print(f"[Cache] done. index={index_path}")
    return index_path


# =============================================================================
# 5) Leakage-safe split utils (subject-group split)
# =============================================================================
def load_index(index_csv: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with open(index_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            out.append(row)
    return out

def make_subject_group_split(
    index_rows: List[Dict[str, str]],
    seed: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_beats_per_record: int = 1,
) -> Dict[str, List[Dict[str, str]]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # filter
    rows = [r for r in index_rows if int(r["num_beats"]) >= int(min_beats_per_record)]
    if len(rows) == 0:
        raise RuntimeError("No records after filtering min_beats_per_record")

    # group by subject_id
    subj2recs: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        subj2recs.setdefault(r["subject_id"], []).append(r)

    subjects = sorted(subj2recs.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = max(1, n_train)
    n_val = max(0, min(n - n_train, n_val))
    n_test = n - n_train - n_val

    train_subj = set(subjects[:n_train])
    val_subj = set(subjects[n_train:n_train + n_val])
    test_subj = set(subjects[n_train + n_val:])

    split = {"train": [], "val": [], "test": []}
    for s in train_subj:
        split["train"].extend(subj2recs[s])
    for s in val_subj:
        split["val"].extend(subj2recs[s])
    for s in test_subj:
        split["test"].extend(subj2recs[s])

    # basic sanity: no overlap
    assert set([r["subject_id"] for r in split["train"]]).isdisjoint(set([r["subject_id"] for r in split["val"]]))
    assert set([r["subject_id"] for r in split["train"]]).isdisjoint(set([r["subject_id"] for r in split["test"]]))
    assert set([r["subject_id"] for r in split["val"]]).isdisjoint(set([r["subject_id"] for r in split["test"]]))

    return split

def save_split_json(split: Dict[str, List[Dict[str, str]]], out_path: str) -> None:
    safe_mkdir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

def load_split_json(path: str) -> Dict[str, List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 6) Shared memmap record store (fast)
# =============================================================================
class MemmapRecordStore:
    """
    Keeps per-record memmap arrays open within each DataLoader worker process.
    This avoids reopening files for every __getitem__ and is typically faster.

    Note: in multi-worker DataLoader, each worker has its own store instance.
    """
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, cache_path: str) -> Dict[str, Any]:
        if cache_path in self._cache:
            return self._cache[cache_path]

        X = np.load(os.path.join(cache_path, "X.npy"), mmap_mode="r")
        r_idx = np.load(os.path.join(cache_path, "r_idx.npy"), mmap_mode="r")
        seg_id = np.load(os.path.join(cache_path, "seg_id.npy"), mmap_mode="r")
        seg_start = np.load(os.path.join(cache_path, "seg_start.npy"), mmap_mode="r")
        seg_len = np.load(os.path.join(cache_path, "seg_len.npy"), mmap_mode="r")
        fs = float(np.load(os.path.join(cache_path, "fs.npy"), mmap_mode="r")[0])

        d = {
            "X": X, "r_idx": r_idx, "seg_id": seg_id,
            "seg_start": seg_start, "seg_len": seg_len, "fs": fs
        }
        self._cache[cache_path] = d
        return d


# =============================================================================
# 7) Negative sampling utilities
# =============================================================================
@dataclass
class NegPoolEntry:
    cache_path: str
    num_beats: int

def build_negative_pool(records: List[Dict[str, str]], min_beats: int = 1) -> List[NegPoolEntry]:
    pool: List[NegPoolEntry] = []
    for r in records:
        nb = int(r["num_beats"])
        if nb >= min_beats:
            pool.append(NegPoolEntry(cache_path=r["cache_path"], num_beats=nb))
    if len(pool) == 0:
        raise RuntimeError("Negative pool is empty after filtering.")
    return pool


# =============================================================================
# 8) Dataset A: InfoNCE CPC (context K -> predict P future beats)
# =============================================================================
class CPCInfoNCEDataset(Dataset):
    """
    Each item:
      x_ctx: (K, win_len) float32
      x_pos: (P, win_len) float32     (future beats, positive targets)
      x_neg: (Nneg, win_len) float32  (optional external negatives; may be empty)
      meta: dict

    Negatives
    - In-batch negatives are preferred (fast). Most CPC implementations use in-batch negatives:
      positives of other samples act as negatives for each sample.
    - If you also want explicit negatives, set n_external_neg > 0.

    Segmentation
    - Sampling is restricted to within a single RR-consistent segment (segmented in cache).
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        K: int = 8,
        P: int = 4,
        stride: int = 1,
        n_external_neg: int = 0,
        neg_pool_records: Optional[List[Dict[str, str]]] = None,
        normalize: str = "per_chunk_z",  # "none"|"per_chunk_z"
        seed: int = 0,
    ):
        self.records = records
        self.K = int(K)
        self.P = int(P)
        self.stride = int(stride)
        self.n_external_neg = int(n_external_neg)
        self.normalize = normalize
        self.rng = np.random.default_rng(seed)

        self.store = MemmapRecordStore()

        # build negative pool from *train records* typically
        if self.n_external_neg > 0:
            base = neg_pool_records if neg_pool_records is not None else records
            self.neg_pool = build_negative_pool(base, min_beats=1)
        else:
            self.neg_pool = []

        # Build index over (record, segment, start_beat_in_segment)
        self.index: List[Tuple[int, int, int]] = []
        need = self.K + self.P
        for ridx, r in enumerate(self.records):
            rec = self.store.get(r["cache_path"])
            seg_start = rec["seg_start"]
            seg_len = rec["seg_len"]
            S = int(seg_start.shape[0])
            for sid in range(S):
                L = int(seg_len[sid])
                if L < need:
                    continue
                s0 = int(seg_start[sid])
                # within segment, we can pick local start t where t+need<=L
                # global start = s0 + t
                for t in range(0, L - need + 1, self.stride):
                    self.index.append((ridx, sid, s0 + t))

        if len(self.index) == 0:
            raise RuntimeError("No valid CPC sequences found. Check K/P or segmentation settings.")

    def __len__(self) -> int:
        return len(self.index)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return x
        if self.normalize == "per_chunk_z":
            mu = np.mean(x, axis=-1, keepdims=True)
            sd = np.std(x, axis=-1, keepdims=True) + 1e-6
            return (x - mu) / sd
        raise ValueError(f"Unknown normalize: {self.normalize}")

    def _sample_external_negs(self, n: int, win_len: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, win_len), dtype=np.float32)
        xs = []
        for _ in range(n):
            entry = self.neg_pool[int(self.rng.integers(0, len(self.neg_pool)))]
            rec = self.store.get(entry.cache_path)
            X = rec["X"]
            b = int(self.rng.integers(0, X.shape[0]))
            xs.append(np.asarray(X[b], dtype=np.float32))
        return np.stack(xs, axis=0).astype(np.float32, copy=False)

    def __getitem__(self, idx: int):
        ridx, sid, start = self.index[idx]
        r = self.records[ridx]
        rec = self.store.get(r["cache_path"])
        X = rec["X"]
        win_len = int(X.shape[1])

        need = self.K + self.P
        seq = np.asarray(X[start:start + need], dtype=np.float32)  # (need, win_len)
        seq = self._norm(seq)

        x_ctx = seq[: self.K]
        x_pos = seq[self.K:self.K + self.P]

        x_neg = self._sample_external_negs(self.n_external_neg, win_len=win_len)
        if x_neg.shape[0] > 0:
            x_neg = self._norm(x_neg)

        meta = {
            "cohort": r["cohort"],
            "record_id": r["record_id"],
            "subject_id": r["subject_id"],
            "segment_id": int(sid),
            "start_beat": int(start),
        }

        return torch.from_numpy(x_ctx), torch.from_numpy(x_pos), torch.from_numpy(x_neg), meta


def cpc_infonce_collate(batch):
    """
    Returns:
      x_ctx: (B, K, W)
      x_pos: (B, P, W)
      x_neg: (B, Nneg, W)
      meta: list[dict]
    """
    x_ctx = torch.stack([b[0] for b in batch], dim=0)
    x_pos = torch.stack([b[1] for b in batch], dim=0)
    x_neg = torch.stack([b[2] for b in batch], dim=0)
    meta = [b[3] for b in batch]
    return x_ctx, x_pos, x_neg, meta


# =============================================================================
# 9) Dataset B: CPC v2 masked prediction (mask beats in a sequence)
# =============================================================================
class CPCv2MaskedDataset(Dataset):
    """
    Each item:
      x_in:   (L, W) float32 with masked positions replaced (e.g., zeros)
      mask:   (L,) bool  True where masked
      x_tgt:  (L, W) float32 original sequence (for masked positions)
      meta: dict

    Mask policy:
    - mask_ratio: fraction of beats to mask in the sequence
    - mask_future_only: if True, mask only positions after context_len (typical CPC-like)
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        seq_len: int = 16,
        stride: int = 1,
        context_len: int = 8,
        mask_ratio: float = 0.25,
        mask_future_only: bool = True,
        mask_value: float = 0.0,
        normalize: str = "per_chunk_z",
        seed: int = 0,
    ):
        assert 0.0 < mask_ratio < 1.0
        assert 1 <= context_len < seq_len

        self.records = records
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.context_len = int(context_len)
        self.mask_ratio = float(mask_ratio)
        self.mask_future_only = bool(mask_future_only)
        self.mask_value = float(mask_value)
        self.normalize = normalize
        self.rng = np.random.default_rng(seed)

        self.store = MemmapRecordStore()

        # index over segments
        self.index: List[Tuple[int, int, int]] = []  # (record_idx, segment_id, start)
        need = self.seq_len
        for ridx, r in enumerate(self.records):
            rec = self.store.get(r["cache_path"])
            seg_start = rec["seg_start"]
            seg_len = rec["seg_len"]
            S = int(seg_start.shape[0])
            for sid in range(S):
                L = int(seg_len[sid])
                if L < need:
                    continue
                s0 = int(seg_start[sid])
                for t in range(0, L - need + 1, self.stride):
                    self.index.append((ridx, sid, s0 + t))

        if len(self.index) == 0:
            raise RuntimeError("No valid masked sequences found. Check seq_len or segmentation settings.")

    def __len__(self) -> int:
        return len(self.index)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return x
        if self.normalize == "per_chunk_z":
            mu = np.mean(x, axis=-1, keepdims=True)
            sd = np.std(x, axis=-1, keepdims=True) + 1e-6
            return (x - mu) / sd
        raise ValueError(f"Unknown normalize: {self.normalize}")

    def __getitem__(self, idx: int):
        ridx, sid, start = self.index[idx]
        r = self.records[ridx]
        rec = self.store.get(r["cache_path"])
        X = rec["X"]

        seq = np.asarray(X[start:start + self.seq_len], dtype=np.float32)  # (L, W)
        seq = self._norm(seq)

        # choose mask positions
        L = self.seq_len
        n_mask = max(1, int(round(L * self.mask_ratio)))

        if self.mask_future_only:
            candidates = np.arange(self.context_len, L)
        else:
            candidates = np.arange(0, L)

        if candidates.size < n_mask:
            n_mask = int(candidates.size)

        mask_pos = self.rng.choice(candidates, size=n_mask, replace=False)
        mask = np.zeros((L,), dtype=bool)
        mask[mask_pos] = True

        x_in = seq.copy()
        x_in[mask] = self.mask_value

        meta = {
            "cohort": r["cohort"],
            "record_id": r["record_id"],
            "subject_id": r["subject_id"],
            "segment_id": int(sid),
            "start_beat": int(start),
        }

        return (
            torch.from_numpy(x_in),
            torch.from_numpy(mask.astype(np.bool_)),
            torch.from_numpy(seq),   # target is original
            meta
        )


def cpc_v2_collate(batch):
    """
    Returns:
      x_in: (B, L, W)
      mask: (B, L)
      x_tgt:(B, L, W)
      meta: list[dict]
    """
    x_in = torch.stack([b[0] for b in batch], dim=0)
    mask = torch.stack([torch.from_numpy(np.asarray(b[1], dtype=np.bool_)) if not torch.is_tensor(b[1]) else b[1] for b in batch], dim=0)
    x_tgt = torch.stack([b[2] for b in batch], dim=0)
    meta = [b[3] for b in batch]
    return x_in, mask, x_tgt, meta


# =============================================================================
# 10) Fast negative sampling strategy recommendation (training-side)
# =============================================================================
"""
InfoNCE best practice:
- Use in-batch negatives for free:
    logits = sim(z_ctx[i], z_pos[j]) over j in batch (and optionally time steps)
- This is faster and usually stronger than explicit external negatives.

The dataset still supports optional external negatives (x_neg), but you can set n_external_neg=0
and rely purely on in-batch negatives for maximum throughput.
"""


# =============================================================================
# 11) CLI entry
# =============================================================================
def main():
    if RUN_BUILD_CACHE:
        if not ROOT_DIR:
            raise ValueError("ROOT_DIR is required for building cache.")
        build_cache(
            root_dir=ROOT_DIR,
            cache_dir=CACHE_DIR,
            pre=PRE_SAMPLES, 
            post=POST_SAMPLES,
            rr_gap_limit_sec=RR_GAP_LIMIT_SEC,
            min_segment_beats=MIN_SEG_BEATS,
            dtype_store=DTYPE_STORE,
            overwrite=OVERWRITE_CACHE,
        )

    if RUN_MAKE_SPLIT:
        index_csv = os.path.join(CACHE_DIR, "index.csv")
        rows = load_index(index_csv)
        split = make_subject_group_split(
            rows, 
            seed=RANDOM_SEED,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            min_beats_per_record=MIN_SEG_BEATS,
        )
        save_split_json(split, SPLIT_JSON_PATH)
        print(f"[Split] saved to {SPLIT_JSON_PATH}")
        print(f"[Split] train recs={len(split['train'])} val recs={len(split['val'])} test recs={len(split['test'])}")

    if RUN_DEMO_LOADER:
        split = load_split_json(SPLIT_JSON_PATH)

        # InfoNCE demo
        ds_infonce = CPCInfoNCEDataset(
            records=split["train"],
            K=8, P=4, stride=1,
            n_external_neg=0,           # prefer in-batch negatives
            neg_pool_records=split["train"],
            normalize="per_chunk_z",
            seed=RANDOM_SEED,
        )
        dl_infonce = DataLoader(
            ds_infonce,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=cpc_infonce_collate,
        )
        x_ctx, x_pos, x_neg, meta = next(iter(dl_infonce))
        print("[InfoNCE Batch]", x_ctx.shape, x_pos.shape, x_neg.shape, meta[0])

        # CPC v2 masked demo
        ds_v2 = CPCv2MaskedDataset(
            records=split["train"],
            seq_len=16, context_len=8,
            stride=1,
            mask_ratio=0.25,
            mask_future_only=True,
            mask_value=0.0,
            normalize="per_chunk_z",
            seed=RANDOM_SEED,
        )
        dl_v2 = DataLoader(
            ds_v2,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=cpc_v2_collate,
        )
        x_in, mask, x_tgt, meta2 = next(iter(dl_v2))
        print("[CPCv2 Batch]", x_in.shape, mask.shape, x_tgt.shape, meta2[0])

# =============================================================================
# 진단 / 검증 함수
# =============================================================================

def diagnose_mat_file(mat_path: str) -> None:
    """
    .mat 파일 내부 구조를 탐색해 Y.npy 누락 원인을 진단.

    출력 항목:
      - 전체 HDF5 키 목록
      - dECG 길이 (ECG 샘플 수 n)
      - final_flag 존재 여부, shape, dtype
      - 3개 실패 구간 각각에 대한 진단 결과
      - 기타 라벨 후보 키
    """
    print(f"\n{'='*60}")
    print(f"[diagnose] {os.path.basename(mat_path)}")
    print(f"{'='*60}")

    if not os.path.isfile(mat_path):
        print(f"  [ERROR] 파일 없음: {mat_path}")
        return

    with h5py.File(mat_path, "r") as f:
        keys = list(f.keys())
        print(f"  HDF5 Keys ({len(keys)}개): {keys}")

        # ECG 길이
        ecg_n = None
        for cand in ["dECG", "ECG", "ecg"]:
            if cand in f:
                arr = np.array(f[cand])
                ecg_n = int(np.prod(arr.shape))
                print(f"  [{cand}] shape={arr.shape}  → 신호 길이 n={ecg_n:,}")
                break
        if ecg_n is None:
            print("  [WARN] ECG 키(dECG) 없음")

        # ── 실패 구간 1: final_flag 키 탐색 ──────────────────────────────────
        ff_key = None
        for k in keys:
            if k.lower() == "final_flag":
                ff_key = k
                break
        if ff_key is None:
            print(f"\n  [✗ 실패 구간 1] 'final_flag' 키 없음 → Y.npy 생성 불가")
            label_hints = [k for k in keys
                           if any(s in k.lower() for s in
                                  ("flag", "label", "class", "annot", "beat"))]
            if label_hints:
                print(f"  [힌트] 라벨 후보 키: {label_hints}")
                print(f"         → BEAT_LABEL_KEY 를 해당 이름으로 변경하세요.")
            return

        ff_raw = np.array(f[ff_key])
        print(f"\n  [✓ 구간 1 통과] '{ff_key}' 발견  shape={ff_raw.shape}  dtype={ff_raw.dtype}")
        nz = int((ff_raw != 0).sum())
        print(f"  non-zero 원소 수: {nz:,}  (전체 {ff_raw.size:,})")

        # ── 실패 구간 2: shape 분석 ──────────────────────────────────────────
        print(f"\n  [구간 2 분석] shape={ff_raw.shape}  ndim={ff_raw.ndim}")
        if ff_raw.ndim == 2:
            r, c = ff_raw.shape
            if r == 27 and c != 27:
                print(f"  → (27, n) 형태: 전치 후 (n=={c}, 27) 으로 사용 가능  ✓")
                n_samples = c
            elif c == 27:
                print(f"  → (n=={r}, 27) 형태: 전치 불필요  ✓")
                n_samples = r
            else:
                print(f"  [✗ 실패 구간 2] shape=({r},{c}) — 행도 열도 27이 아님")
                print(f"  → 실제 클래스 수({min(r,c)})를 확인하고 코드의 '27' 조건을 수정하세요.")
                return
        else:
            print(f"  [✗ 실패 구간 2] ndim={ff_raw.ndim} ≠ 2 — shape={ff_raw.shape}")
            print(f"  → 1차원(클래스 인덱스) 또는 3D+ 배열로 저장된 경우 별도 파싱 필요")
            return

        # ── 실패 구간 3: 길이 불일치 ─────────────────────────────────────────
        print(f"\n  [구간 3 분석] final_flag n_samples={n_samples:,}  ECG n={ecg_n:,}")
        if ecg_n is not None and n_samples != ecg_n:
            diff = abs(n_samples - ecg_n)
            pct  = diff / ecg_n * 100
            print(f"  [✗ 실패 구간 3] 길이 불일치: 차이={diff:,} 샘플 ({pct:.2f}%)")
            print(f"  → 리샘플링/패딩 여부 확인 필요")
        else:
            print(f"  [✓ 구간 3 통과] 길이 일치")
            print(f"\n  ★ 모든 구간 통과. 실제 로드 확인:")
            print(f"     load_holter_mat_v73_minimal('{mat_path}', verbose=True)")

    print(f"{'='*60}\n")


def verify_label_extraction(mat_path: str, pre: int = 250, post: int = 250):
    """
    라벨 추출 및 비트 매핑 알고리즘이 정상 작동하는지 확인하기 위해,
    디스크 저장 없이 메모리 상에서 X, Y의 5개 샘플만 출력하는 검증 함수입니다.
    """
    print(f"=== [검증 시작] {mat_path} 로드 중 ===")
    try:
        # 1. 데이터 로드 (기존 함수 재사용)
        rec = load_holter_mat_v73_minimal(mat_path, ecg_mv_dtype=np.float32, verbose=True)
        print(f"- 원본 ECG 데이터 길이: {rec.n}")
        
        if rec.final_flag is not None:
            print(f"- 원본 라벨(final_flag) 형상: {rec.final_flag.shape}")
        else:
            print("- 경고: 파일 내에 final_flag가 존재하지 않습니다.")

        # 2. 유효성 마스크 생성
        valid = build_valid_mask_from_quality(rec, exclude_data_lost=True, exclude_lead_off=True)

        # 3. 비트 및 라벨 동시 추출
        X, r_idx, valid_ratio, Y = extract_beats_from_record(
            rec, valid_mask=valid, pre=pre, post=post,
            min_valid_ratio=1.0, use_only_valid_rpeaks=True
        )

        # 4. 검증 결과 출력 (최대 5개 샘플)
        num_samples = min(5, X.shape[0])
        print(f"\n=== [추출 결과 확인] 총 {X.shape[0]}개 비트 중 상위 {num_samples}개 출력 ===")

        if num_samples == 0:
            print("추출된 유효 비트가 없습니다. (R-peak가 없거나 모두 노이즈 구간임)")
            return

        print(f"[1] 추출된 신호 (X) 형상: {X[:num_samples].shape} -> (Batch, Window_Length)")
        # 데이터가 너무 길어 앞 5개 수치만 출력
        print(f"    X 데이터 샘플 (첫 5개 샘플의 앞 5개 값): \n{X[:num_samples, :5]}")

        if Y is not None:
            print(f"\n[2] 추출된 라벨 (Y) 형상: {Y[:num_samples].shape} -> (Batch, 27 Classes)")
            print(f"    Y 데이터 샘플 (각 비트당 27차원 라벨 벡터): \n{Y[:num_samples]}")
        else:
            print("\n[2] 라벨(Y) 데이터가 추출되지 않아 출력을 건너뜁니다.")

    except Exception as e:
        print(f"검증 중 오류 발생: {e}")

import os
import glob
import numpy as np
import matplotlib
# [중요] GUI가 없는 서버 환경에서 오류 없이 이미지로만 저장하기 위한 백엔드 설정
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def save_ecg_sample_plots(mat_paths: list, 
                          save_dir: str = "/home/young00/ECG_tf/test_samples/", 
                          num_files: int = 5, 
                          duration_sec: float = 10.0):
    """
    서버 환경에서 .mat 파일의 초기 구간(duration_sec)을 Plot하여 이미지로 저장합니다.
    실제 R-peak 위치에 붉은 점과 라벨(클래스 번호)을 표기합니다.
    """
    # 저장할 디렉토리가 없으면 생성
    os.makedirs(save_dir, exist_ok=True)
    
    files_to_plot = mat_paths[:num_files]
    if not files_to_plot:
        print("시각화할 파일이 리스트에 없습니다.")
        return

    print(f"\n=== [시각화 시작] {len(files_to_plot)}개 파일에 대해 {duration_sec}초간 신호 이미지 저장 ===")
    
    for mat_path in files_to_plot:
        filename = os.path.basename(mat_path)
        save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_plot.png")
        
        try:
            # 1. 데이터 로드 (기존에 정의된 함수 사용)
            rec = load_holter_mat_v73_minimal(mat_path, ecg_mv_dtype=np.float32)
            
            # 2. OOM 방지용 구간 자르기 (초 단위)
            max_samples = min(int(duration_sec * rec.fs), rec.n)
            
            time_axis = np.arange(max_samples) / rec.fs
            ecg_slice = rec.ecg_mv[:max_samples]
            rpeak_slice = rec.rpeak[:max_samples]
            
            # 3. R-peak 인덱스 추출
            rp_indices = np.where(rpeak_slice > 0)[0]
            
            # 4. Figure 생성 (크기: 가로 18인치, 세로 4인치)
            fig, ax = plt.subplots(figsize=(18, 4))
            
            # 베이스 심전도 신호 그리기
            ax.plot(time_axis, ecg_slice, color='black', linewidth=1.0, alpha=0.8, label='ECG (mV)')
            
            # R-peak 마커 및 라벨 텍스트 처리
            for rp_idx in rp_indices:
                # 붉은 점 표기
                ax.plot(time_axis[rp_idx], ecg_slice[rp_idx], 'ro', markersize=6)
                
                label_text = "R"
                if rec.final_flag is not None:
                    # 27차원 벡터 중 1로 활성화된 인덱스 탐색
                    active_classes = np.where(rec.final_flag[rp_idx] > 0)[0]
                    if len(active_classes) > 0:
                        label_text = f"C:{','.join(map(str, active_classes))}"
                    else:
                        label_text = "C:None"
                
                # 텍스트 주석 달기
                ax.annotate(label_text, 
                            xy=(time_axis[rp_idx], ecg_slice[rp_idx]), 
                            xytext=(0, 10), 
                            textcoords="offset points", 
                            ha='center', va='bottom', 
                            fontsize=10, color='blue', fontweight='bold', rotation=45)
            
            ax.set_title(f"ECG Signal - {filename} (First {duration_sec}s)")
            ax.set_xlabel("Time (Seconds)")
            ax.set_ylabel("Amplitude (mV)")
            ax.legend(loc="upper right")
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # 5. 디스크에 이미지 저장 및 메모리 해제
            plt.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig) # ★매우 중요: 루프 내 메모리 누수 방지
            
            print(f"- 저장 완료: {save_path}")
            
        except Exception as e:
            print(f"- [실패] {filename} 시각화 중 오류 발생: {e}")


if __name__ == "__main__":
    if RUN_MODE == "build":
        main()

    elif RUN_MODE == "diagnose":
        # ── 3개 실패 구간 자동 탐지 ──────────────────────────────────────────
        diagnose_mat_file(DIAGNOSE_MAT_PATH)

    elif RUN_MODE == "verify":
        # ── 메모리 검증 (디스크 저장 없음) ───────────────────────────────────
        verify_label_extraction(DIAGNOSE_MAT_PATH, pre=PRE_SAMPLES, post=POST_SAMPLES)

    elif RUN_MODE == "plot":
        # ── ECG + R-peak 시각화 이미지 저장 ──────────────────────────────────
        target_dir = os.path.dirname(DIAGNOSE_MAT_PATH)
        mat_files  = sorted(glob.glob(os.path.join(target_dir, "*.mat")))
        save_ecg_sample_plots(
            mat_paths=mat_files,
            save_dir=PLOT_SAVE_DIR,
            num_files=PLOT_NUM_FILES,
            duration_sec=PLOT_DURATION_SEC,
        )

    else:
        raise ValueError(f"RUN_MODE 는 'build'/'diagnose'/'verify'/'plot' 중 하나여야 합니다: {RUN_MODE!r}")