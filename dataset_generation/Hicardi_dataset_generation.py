# -*- coding: utf-8 -*-
"""
Holter Beat Pretraining Pipeline (Fully Parameterized Generator)

Implemented requirements
1. Standalone load_holter_mat_v73_minimal & discover_records.
2. ADC-to-mV conversion decoupled from loader, implemented as Generator option.
3. ALL windowing and filter options exposed as explicit Class Arguments.
4. Restored argparse for external CLI control without code modification.
5. Removed model-specific (CPC) terminologies for general representation learning.
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
import matplotlib

# GUI가 없는 서버 환경에서 오류 없이 이미지로만 저장하기 위한 백엔드 설정
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

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

def extract_subject_id_default(cohort: str, record_id: str) -> str:
    rid = record_id
    for sep in ["_", "-"]:
        if sep in rid:
            rid = rid.split(sep)[0]
            break
    return f"{cohort}::{rid}"


# =============================================================================
# 1) Standalone Functions
# =============================================================================
@dataclass
class HolterRecord:
    fs: float
    ecg_signal: np.ndarray  
    rpeak: np.ndarray       
    data_lost: np.ndarray   
    lead_off: np.ndarray    
    final_flag: Optional[np.ndarray] = None  

    @property
    def n(self) -> int:
        return int(self.ecg_signal.shape[0])

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

def load_holter_mat_v73_minimal(mat_path: str, verbose: bool = False) -> HolterRecord:
    """
    Load raw ECG data from MATLAB -v7.3 HDF5. 
    Does NOT convert to mV. Returns raw ADC values.
    """
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(mat_path)

    with h5py.File(mat_path, "r") as f:
        def get_obj(name: str):
            if name in f: return f[name]
            for k in f.keys():
                if k.lower() == name.lower(): return f[k]
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
            raise KeyError("R-peak variable not found.")

        _final_flag_raw = None
        try:
            ff = np.array(get_obj("final_flag"))
            if ff.ndim == 2:
                if ff.shape[0] == 27 and ff.shape[1] != 27:
                    ff = ff.T
                if ff.shape[1] == 27:
                    ff = np.where(ff > 0, 1, 0)
                    _final_flag_raw = ff.astype(np.int8, copy=False)
        except KeyError:
            pass

    ecg_adc = np.nan_to_num(_as_1d(dECG), nan=8192.0, posinf=8192.0, neginf=8192.0).astype(np.float32, copy=False)
    rpeak = np.nan_to_num(_as_1d(rpk_flag), nan=0.0).astype(np.int8, copy=False)
    data_lost_1d = np.nan_to_num(_as_1d(data_lost), nan=1.0).astype(np.int8, copy=False)
    lead_off_1d = np.nan_to_num(_as_1d(lead_off), nan=1.0).astype(np.int8, copy=False)
    
    n = ecg_adc.shape[0]
    if _final_flag_raw is not None and _final_flag_raw.shape[0] != n:
        _final_flag_raw = None

    return HolterRecord(fs=fs, ecg_signal=ecg_adc, rpeak=rpeak, data_lost=data_lost_1d, lead_off=lead_off_1d, final_flag=_final_flag_raw)


# =============================================================================
# 2) Generator Pipeline Class (Fully Parameterized)
# =============================================================================
class HicardiCacheGenerator:
    """
    Generator class with fully explicit arguments for windowing and signal processing.
    No hidden internal dictionaries.
    """
    def __init__(
        self,
        root_dir: str,
        cache_dir: str,
        pre: int = 250,
        post: int = 250,
        min_valid_ratio: float = 1.0,
        use_only_valid_rpeaks: bool = True,
        exclude_data_lost: bool = True,
        exclude_lead_off: bool = True,
        rr_gap_limit_sec: float = 2.0,
        min_segment_beats: int = 4,
        convert_to_mv: bool = True,
        adc_offset: float = 8192.0,
        adc_scale: float = 1000.0,
        dtype_store: str = "float32",
        overwrite: bool = False
    ):
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.pre = pre
        self.post = post
        self.min_valid_ratio = min_valid_ratio
        self.use_only_valid_rpeaks = use_only_valid_rpeaks
        self.exclude_data_lost = exclude_data_lost
        self.exclude_lead_off = exclude_lead_off
        self.rr_gap_limit_sec = rr_gap_limit_sec
        self.min_segment_beats = min_segment_beats
        self.convert_to_mv = convert_to_mv
        self.adc_offset = adc_offset
        self.adc_scale = adc_scale
        self.dtype_store = dtype_store
        self.overwrite = overwrite

    def generate(self) -> str:
        safe_mkdir(self.cache_dir)
        index_path = os.path.join(self.cache_dir, "index.csv")
        records = discover_records(self.root_dir, exts=(".mat",))
        print(f"[CacheGenerator] Discovered records: {len(records)}")

        rows = []
        for cohort, record_id, record_path in records:
            out_dir = os.path.join(self.cache_dir, "cohorts", cohort, record_id)
            row = self._process_single_record(cohort, record_id, record_path, out_dir)
            if row:
                rows.append(row)

        self._write_index_csv(index_path, rows)
        print(f"[CacheGenerator] Done. Index saved to {index_path}")
        return index_path

    def _process_single_record(self, cohort: str, record_id: str, record_path: str, out_dir: str) -> Optional[List[str]]:
        if (not self.overwrite) and os.path.isfile(os.path.join(out_dir, "X.npy")):
            try:
                X_shape = np.load(os.path.join(out_dir, "X.npy"), mmap_mode="r").shape
                seg_start = np.load(os.path.join(out_dir, "seg_start.npy"), mmap_mode="r")
                subject_id = extract_subject_id_default(cohort, record_id)
                return [cohort, record_id, subject_id, record_path, out_dir, str(int(X_shape[0])), str(int(seg_start.shape[0]))]
            except Exception:
                pass
            return None

        try:
            rec = load_holter_mat_v73_minimal(record_path)
            self._apply_mv_conversion(rec)
            valid_mask = self._build_valid_mask(rec)
            
            X, r_idx, _, Y = self._extract_beats(rec, valid_mask)
            if X.shape[0] == 0: return None

            seg_id, seg_start, seg_len = self._segment_beats(r_idx, rec.fs)
            
            keep = seg_id >= 0
            X, r_idx, seg_id = X[keep], r_idx[keep], seg_id[keep]
            if Y is not None: Y = Y[keep]

            if X.shape[0] == 0: return None

            sid = seg_id.astype(np.int32)
            changes = np.where(np.diff(sid, prepend=sid[0]) != 0)[0]
            starts2 = changes.astype(np.int32)
            if starts2.size == 0 or starts2[0] != 0:
                starts2 = np.insert(starts2, 0, 0).astype(np.int32)
            lens2 = (np.append(starts2[1:], np.array([sid.shape[0]], dtype=np.int32)) - starts2).astype(np.int32)

            self._write_cache_to_disk(out_dir, X, r_idx, sid, starts2, lens2, rec.fs, Y)

            subject_id = extract_subject_id_default(cohort, record_id)
            print(f"[CacheGenerator] {cohort}/{record_id}: beats={X.shape[0]} segs={starts2.shape[0]}")
            return [cohort, record_id, subject_id, record_path, out_dir, str(int(X.shape[0])), str(int(starts2.shape[0]))]

        except Exception as e:
            print(f"[CacheGenerator][FAIL] {record_path} :: {type(e).__name__}: {e}")
            return None

    def _apply_mv_conversion(self, rec: HolterRecord) -> None:
        if self.convert_to_mv:
            rec.ecg_signal = (rec.ecg_signal - self.adc_offset) / self.adc_scale

    def _build_valid_mask(self, rec: HolterRecord) -> np.ndarray:
        valid = np.ones(rec.n, dtype=bool)
        if self.exclude_data_lost:
            valid &= (rec.data_lost == 0)
        if self.exclude_lead_off:
            valid &= (rec.lead_off == 0)
        return valid

    def _extract_beats(self, rec: HolterRecord, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        win_len = self.pre + self.post + 1
        has_labels = rec.final_flag is not None

        rp = np.where(rec.rpeak > 0)[0]
        if rp.size == 0:
            return (np.zeros((0, win_len), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32), np.zeros((0, 27), np.int8) if has_labels else None)

        if self.use_only_valid_rpeaks:
            rp = rp[valid_mask[rp]]

        X_list, ridx_list, vratio_list, Y_list = [], [], [], []

        for r in rp:
            s = int(r - self.pre)
            e = int(r + self.post + 1)
            if s < 0 or e > rec.n: continue
            
            x = rec.ecg_signal[s:e]
            if x.shape[0] != win_len: continue

            vratio = float(np.mean(valid_mask[s:e]))
            if vratio < self.min_valid_ratio: continue

            X_list.append(x.astype(np.float32, copy=False))
            ridx_list.append(int(r))
            vratio_list.append(vratio)
            if has_labels: Y_list.append(rec.final_flag[int(r)])

        if not X_list:
            return (np.zeros((0, win_len), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32), np.zeros((0, 27), np.int8) if has_labels else None)

        X = np.stack(X_list, axis=0)
        r_idx = np.asarray(ridx_list, dtype=np.int32)
        valid_ratio = np.asarray(vratio_list, dtype=np.float32)
        Y = np.stack(Y_list, axis=0).astype(np.int8, copy=False) if has_labels else None
        
        return X, r_idx, valid_ratio, Y

    def _segment_beats(self, r_idx: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        B = int(r_idx.shape[0])
        if B == 0: return np.zeros((0,), np.int32), np.zeros((0,), np.int32), np.zeros((0,), np.int32)

        gaps = np.diff(r_idx.astype(np.int64)) / float(fs)
        cut = np.zeros((B,), dtype=bool)
        cut[0] = True
        if B > 1:
            cut[1:] = gaps <= float(self.rr_gap_limit_sec)
            starts = [0] + [i for i in range(1, B) if not cut[i]]
        else:
            starts = [0]

        starts = np.asarray(starts, dtype=np.int32)
        ends = np.append(starts[1:], np.array([B], dtype=np.int32))
        lens = (ends - starts).astype(np.int32)

        keep = lens >= int(self.min_segment_beats)
        starts_k, lens_k = starts[keep], lens[keep]

        seg_id = np.full((B,), -1, dtype=np.int32)
        for sid, (s, ln) in enumerate(zip(starts_k.tolist(), lens_k.tolist())):
            seg_id[s:s+ln] = sid

        return seg_id, starts_k, lens_k

    def _write_cache_to_disk(self, out_dir: str, X: np.ndarray, r_idx: np.ndarray, seg_id: np.ndarray, seg_start: np.ndarray, seg_len: np.ndarray, fs: float, Y: Optional[np.ndarray]) -> None:
        safe_mkdir(out_dir)
        X_store = X.astype(np.float16 if self.dtype_store == "float16" else np.float32, copy=False)
        np.save(os.path.join(out_dir, "X.npy"), X_store, allow_pickle=False)
        np.save(os.path.join(out_dir, "r_idx.npy"), r_idx.astype(np.int32, copy=False), allow_pickle=False)
        np.save(os.path.join(out_dir, "seg_id.npy"), seg_id.astype(np.int32, copy=False), allow_pickle=False)
        np.save(os.path.join(out_dir, "seg_start.npy"), seg_start.astype(np.int32, copy=False), allow_pickle=False)
        np.save(os.path.join(out_dir, "seg_len.npy"), seg_len.astype(np.int32, copy=False), allow_pickle=False)
        np.save(os.path.join(out_dir, "fs.npy"), np.asarray([fs], dtype=np.float32), allow_pickle=False)
        if Y is not None:
            np.save(os.path.join(out_dir, "Y.npy"), Y.astype(np.int8, copy=False), allow_pickle=False)

    def _write_index_csv(self, index_path: str, rows: List[List[str]]) -> None:
        safe_mkdir(os.path.dirname(index_path) or ".")
        with open(index_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["cohort", "record_id", "subject_id", "record_path", "cache_path", "num_beats", "num_segments"])
            for r in rows: w.writerow(r)


# =============================================================================
# 3) Leakage-safe split utils
# =============================================================================
def load_index(index_csv: str) -> List[Dict[str, str]]:
    with open(index_csv, "r", encoding="utf-8") as f: return list(csv.DictReader(f))

def make_subject_group_split(index_rows: List[Dict[str, str]], seed: int = 0, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, min_beats_per_record: int = 1) -> Dict[str, List[Dict[str, str]]]:
    rows = [r for r in index_rows if int(r["num_beats"]) >= int(min_beats_per_record)]
    subj2recs: Dict[str, List[Dict[str, str]]] = {}
    for r in rows: subj2recs.setdefault(r["subject_id"], []).append(r)
    subjects = sorted(subj2recs.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(0, min(n - n_train, int(round(n * val_ratio))))
    
    split = {"train": [], "val": [], "test": []}
    for s in set(subjects[:n_train]): split["train"].extend(subj2recs[s])
    for s in set(subjects[n_train:n_train + n_val]): split["val"].extend(subj2recs[s])
    for s in set(subjects[n_train + n_val:]): split["test"].extend(subj2recs[s])
    return split

def save_split_json(split: Dict[str, List[Dict[str, str]]], out_path: str) -> None:
    safe_mkdir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding="utf-8") as f: json.dump(split, f, ensure_ascii=False, indent=2)

def load_split_json(path: str) -> Dict[str, List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8") as f: return json.load(f)


# =============================================================================
# 4) PyTorch Datasets (General Sequence & Masking)
# =============================================================================
class MemmapRecordStore:
    def __init__(self): self._cache = {}
    def get(self, cache_path: str) -> Dict[str, Any]:
        if cache_path in self._cache: return self._cache[cache_path]
        d = {
            "X": np.load(os.path.join(cache_path, "X.npy"), mmap_mode="r"),
            "r_idx": np.load(os.path.join(cache_path, "r_idx.npy"), mmap_mode="r"),
            "seg_id": np.load(os.path.join(cache_path, "seg_id.npy"), mmap_mode="r"),
            "seg_start": np.load(os.path.join(cache_path, "seg_start.npy"), mmap_mode="r"),
            "seg_len": np.load(os.path.join(cache_path, "seg_len.npy"), mmap_mode="r"),
            "fs": float(np.load(os.path.join(cache_path, "fs.npy"), mmap_mode="r")[0])
        }
        self._cache[cache_path] = d
        return d

class ECGSequenceDataset(Dataset):
    def __init__(self, records: List[Dict[str, str]], context_len: int = 8, target_len: int = 4, stride: int = 1, normalize: str = "per_chunk_z", seed: int = 0):
        self.records, self.context_len, self.target_len, self.stride, self.normalize = records, context_len, target_len, stride, normalize
        self.rng, self.store = np.random.default_rng(seed), MemmapRecordStore()
        self.index = []
        for ridx, r in enumerate(self.records):
            rec = self.store.get(r["cache_path"])
            for sid in range(int(rec["seg_start"].shape[0])):
                L = int(rec["seg_len"][sid])
                if L >= (context_len + target_len):
                    s0 = int(rec["seg_start"][sid])
                    for t in range(0, L - (context_len + target_len) + 1, self.stride):
                        self.index.append((ridx, sid, s0 + t))

    def __len__(self) -> int: return len(self.index)
    def __getitem__(self, idx: int):
        ridx, sid, start = self.index[idx]
        r = self.records[ridx]
        seq = np.asarray(self.store.get(r["cache_path"])["X"][start:start + self.context_len + self.target_len], dtype=np.float32)
        if self.normalize == "per_chunk_z":
            seq = (seq - np.mean(seq, axis=-1, keepdims=True)) / (np.std(seq, axis=-1, keepdims=True) + 1e-6)
        x_neg = np.zeros((0, seq.shape[1]), dtype=np.float32)
        meta = {"cohort": r["cohort"], "record_id": r["record_id"], "subject_id": r["subject_id"], "segment_id": sid, "start_beat": start}
        return torch.from_numpy(seq[:self.context_len]), torch.from_numpy(seq[self.context_len:]), torch.from_numpy(x_neg), meta

def ecg_sequence_collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]), torch.stack([b[2] for b in batch]), [b[3] for b in batch]


# =============================================================================
# 5) CLI / Main Execution
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Hicardi ECG Pretraining Pipeline")
    
    # Modes
    ap.add_argument("--build_cache", action="store_true", help="Build the cache from raw .mat files")
    ap.add_argument("--make_split", action="store_true", help="Generate train/val/test split json")
    ap.add_argument("--demo_loader", action="store_true", help="Run DataLoader demo")
    
    # Paths
    ap.add_argument("--root", type=str, default="Database/hicardi", help="Root directory containing dataset")
    ap.add_argument("--cache", type=str, default="Database/hicardi_beat_cache", help="Output directory for cache")
    ap.add_argument("--split", type=str, default="Database/hicardi_beat_cache/split.json", help="Path to split.json")
    
    # Generator Parameters
    ap.add_argument("--pre", type=int, default=250, help="Samples before R-peak")
    ap.add_argument("--post", type=int, default=250, help="Samples after R-peak")
    ap.add_argument("--min_valid_ratio", type=float, default=1.0, help="Minimum valid signal ratio")
    ap.add_argument("--rr_gap", type=float, default=2.0, help="RR gap limit in seconds for segmentation")
    ap.add_argument("--min_seg_beats", type=int, default=4, help="Minimum beats per segment")
    ap.add_argument("--dtype_store", type=str, default="float32", choices=["float16", "float32"], help="Cache precision")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing cache")

    # Split Parameters
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    args = ap.parse_args()

    if args.build_cache:
        print("=== Starting Cache Generation ===")
        generator = HicardiCacheGenerator(
            root_dir=args.root,
            cache_dir=args.cache,
            pre=args.pre,
            post=args.post,
            min_valid_ratio=args.min_valid_ratio,
            use_only_valid_rpeaks=True,
            exclude_data_lost=True,
            exclude_lead_off=True,
            rr_gap_limit_sec=args.rr_gap,
            min_segment_beats=args.min_seg_beats,
            convert_to_mv=True,
            dtype_store=args.dtype_store,
            overwrite=args.overwrite
        )
        generator.generate()

    if args.make_split:
        print("=== Generating Data Split ===")
        index_csv = os.path.join(args.cache, "index.csv")
        rows = load_index(index_csv)
        split = make_subject_group_split(
            rows, 
            seed=args.seed, 
            train_ratio=args.train_ratio, 
            val_ratio=args.val_ratio, 
            test_ratio=args.test_ratio, 
            min_beats_per_record=args.min_seg_beats
        )
        save_split_json(split, args.split)
        print(f"[Split] saved to {args.split}")

    if args.demo_loader:
        print("=== Running Demo Loader ===")
        split = load_split_json(args.split)
        ds_seq = ECGSequenceDataset(records=split["train"])
        dl_seq = DataLoader(ds_seq, batch_size=32, collate_fn=ecg_sequence_collate)
        print("[Sequence Batch Loaded]:", next(iter(dl_seq))[0].shape)

if __name__ == "__main__":
    main()