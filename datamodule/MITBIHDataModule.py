# -*- coding: utf-8 -*-
"""
===============================================================================
MIT-BIH DataModule for DANN (with patient IDs + extra features)
-------------------------------------------------------------------------------
- Loads preprocessed MIT-BIH data (train/test .npy)
- Loads extra handcrafted features (train_feats.npy / test_feats.npy)
- Supports optional windowing (Gaussian / Tukey)
- Returns (x, y, pid, feats) tuples for DANN
===============================================================================
"""

import os
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

from .utils.dataModuleUtils import (
    UndersamplingConfig, EfficiencyConfig, DataLoaderConfig, SamplingStrategy,
)
from .utils.io_utils import load_json


# -------------------------------------------------------------------------
# Window functions
# -------------------------------------------------------------------------
def gaussian_window(N: int, std_ratio: float = 0.2, device=None, dtype=torch.float32):
    if device is None:
        device = torch.device("cpu")
    n = torch.arange(N, device=device, dtype=dtype)
    center = (N - 1) / 2.0
    sigma = std_ratio * N
    w = torch.exp(-0.5 * ((n - center) / sigma) ** 2)
    w /= w.max()
    return w


def tukey_window(N: int, alpha: float = 0.2, device=None, dtype=torch.float32):
    if device is None:
        device = torch.device("cpu")
    n = torch.arange(N, device=device, dtype=dtype)
    w = torch.ones(N, device=device, dtype=dtype)
    if alpha <= 0:
        return w
    if alpha >= 1:
        return 0.5 * (1 - torch.cos(2 * torch.pi * n / (N - 1)))
    edge = alpha * (N - 1) / 2.0
    left = n < edge
    right = n > (N - 1 - edge)
    w[left] = 0.5 * (1 + torch.cos(torch.pi * (2 * n[left] / (alpha * (N - 1)) - 1)))
    w[right] = 0.5 * (1 + torch.cos(torch.pi * (2 * n[right] / (alpha * (N - 1)) - 2 / alpha + 1)))
    w /= w.max()
    return w


# -------------------------------------------------------------------------
# Label mapping
# -------------------------------------------------------------------------
def label2index(label):
    m = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
    return m[label]


# -------------------------------------------------------------------------
# Dataset with patient IDs + features
# -------------------------------------------------------------------------
class MITBIHDANN_Dataset(Dataset):
    """Dataset returning (x, y, pid, feats) for DANN"""

    def __init__(self, data, labels, patients, feats,
                 window_type=None, window_param: float = 0.2):

        self.data = data              # [N, 1, L]
        self.labels = labels          # [N]
        self.patients = patients      # [N]
        self.feats = feats            # [N, F]

        # Normalize patient IDs → integer indices
        unique_pids = sorted(np.unique(self.patients))
        self.pid2idx = {pid: i for i, pid in enumerate(unique_pids)}
        self.pid_idx = np.array([self.pid2idx[p] for p in self.patients], dtype=np.int64)

        # Setup window
        self.window = None
        if window_type is not None:
            N = data.shape[-1]
            if window_type.lower() == "gaussian":
                self.window = gaussian_window(N, std_ratio=window_param)
            elif window_type.lower() == "tukey":
                self.window = tukey_window(N, alpha=window_param)
            else:
                raise ValueError(f"Unsupported window type: {window_type}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Raw waveform
        x = torch.tensor(self.data[index], dtype=torch.float32)  # [1, L]
        y = torch.tensor(self.labels[index], dtype=torch.long)
        pid = torch.tensor(self.pid_idx[index], dtype=torch.long)

        # Extra handcrafted features
        feats = torch.tensor(self.feats[index], dtype=torch.float32)  # [F]

        # Apply window only to waveform
        if self.window is not None:
            w = self.window.to(x.device, x.dtype)
            if x.ndim == 2:
                x = x * w.unsqueeze(0)
            else:
                x = x * w

        return x, y, pid, feats


# -------------------------------------------------------------------------
# Lightning DataModule
# -------------------------------------------------------------------------
class MITBIH_DANN_DataModule(L.LightningDataModule):

    def __init__(
        self,
        # HiCardi-compatible interface
        split_json:             Optional[str]            = None,
        return_arg:             Mapping[str, Any]        = None,
        undersampling_arg:      Mapping[str, Any]        = None,
        efficient_training_arg: Mapping[str, Any]        = None,
        dataloader_arg:         Mapping[str, Any]        = None,
        # Legacy interface (backward compat)
        base_dir:               Optional[str]            = None,
        batch_size:             int                      = 128,
        num_workers:            int                      = 8,
        window_type:            Optional[str]            = None,
        window_param:           float                    = 0.2,
        target_win_len:         Optional[int]            = None,
        per_beat_normalize:     bool                     = True,
        **kwargs,
    ):
        super().__init__()

        # base_dir 우선순위: split_json > return_arg["cache_root"] > base_dir param
        if split_json is not None:
            meta = load_json(split_json)
            self.base_dir = meta.get("base_dir", base_dir)
        elif return_arg is not None:
            self.base_dir = (return_arg.get("cache_root")
                             or return_arg.get("base_dir")
                             or base_dir)
        else:
            self.base_dir = base_dir

        if self.base_dir is None:
            raise ValueError(
                "base_dir must be provided via split_json({'base_dir':...}), "
                "return_arg['cache_root'], or the base_dir parameter."
            )

        # 공통 유틸에서 Config 재사용
        self.und_cfg = UndersamplingConfig.from_dict(undersampling_arg or {})
        self.eff_cfg = EfficiencyConfig.from_dict(efficient_training_arg or {})

        # DataLoader 설정: dataloader_arg 우선, 없으면 legacy params
        if dataloader_arg is not None:
            _ldr = DataLoaderConfig.from_dict(dataloader_arg)
            self.batch_size  = _ldr.batch_size
            self.num_workers = _ldr.num_workers
            self._pin_memory = _ldr.pin_memory
            self._drop_last  = _ldr.drop_last
        else:
            self.batch_size  = batch_size
            self.num_workers = num_workers
            self._pin_memory = True
            self._drop_last  = True

        # return_arg 에서 MITBIH 관련 옵션 추출
        _ra = return_arg or {}
        self.n_classes = int(_ra.get("n_classes", 5))

        # MITBIH 고유 파라미터
        self.window_type        = window_type
        self.window_param       = window_param
        self.target_win_len     = int(target_win_len) if target_win_len is not None else None
        self.per_beat_normalize = bool(per_beat_normalize)

    # ── 공통 유틸 재사용: 언더샘플링 + 데이터 이피션트 ─────────────────────────

    def _apply_undersampling(self, data, labels, patients, feats):
        """UndersamplingConfig 재사용 — numpy 배열에 random undersampling 적용."""
        if not self.und_cfg.applies_to("train"):
            return data, labels, patients, feats

        normal_idx   = np.where(labels == 0)[0]   # AAMI N 클래스
        abnormal_idx = np.where(labels != 0)[0]   # S, V, F, Q

        if len(normal_idx) == 0 or len(abnormal_idx) == 0:
            return data, labels, patients, feats

        n_keep = min(len(normal_idx),
                     max(1, int(len(abnormal_idx) * self.und_cfg.normal_ratio)))
        if n_keep >= len(normal_idx):
            return data, labels, patients, feats

        rng          = np.random.default_rng(self.und_cfg.seed)
        kept_normal  = rng.choice(normal_idx, n_keep, replace=False)
        idx          = np.sort(np.concatenate([kept_normal, abnormal_idx]))

        print(f"[MITBIH] Undersampling: normal {len(normal_idx):,} → {n_keep:,} "
              f"(ratio={n_keep / max(len(abnormal_idx), 1):.2f}:1)")
        return data[idx], labels[idx], patients[idx], feats[idx]

    def _apply_efficiency(self, data, labels, patients, feats):
        """EfficiencyConfig 재사용 — variation_factor 서브샘플링 적용."""
        if not self.eff_cfg.applies_to("train"):
            return data, labels, patients, feats

        n      = len(data)
        n_keep = max(1, int(n * self.eff_cfg.variation_factor))
        rng    = np.random.default_rng(self.eff_cfg.seed)
        idx    = np.sort(rng.choice(n, n_keep, replace=False))

        print(f"[MITBIH] Efficiency: {n:,} → {n_keep:,} "
              f"(vf={self.eff_cfg.variation_factor:.2f})")
        return data[idx], labels[idx], patients[idx], feats[idx]

    # ── setup ────────────────────────────────────────────────────────────────

    def setup(self, stage=None):
        train_dir = os.path.join(self.base_dir, "train")
        test_dir = os.path.join(self.base_dir, "test")

        # Load raw waveform
        train_data = np.load(os.path.join(train_dir, "train_data.npy"), allow_pickle=True)
        test_data = np.load(os.path.join(test_dir, "test_data.npy"), allow_pickle=True)

        # Load labels
        train_labels = np.load(os.path.join(train_dir, "train_labels.npy"), allow_pickle=True)
        test_labels = np.load(os.path.join(test_dir, "test_labels.npy"), allow_pickle=True)

        # Load patient IDs
        train_patients = np.load(os.path.join(train_dir, "train_patients.npy"), allow_pickle=True)
        test_patients = np.load(os.path.join(test_dir, "test_patients.npy"), allow_pickle=True)

        # Load features (NEW)
        train_feats = np.load(os.path.join(train_dir, "train_feats.npy"), allow_pickle=True)
        test_feats = np.load(os.path.join(test_dir, "test_feats.npy"), allow_pickle=True)

        # Label to index
        train_labels = np.array([label2index(l) for l in train_labels], dtype=np.int64)
        test_labels = np.array([label2index(l) for l in test_labels], dtype=np.int64)

        # Expand channels if needed
        if train_data.ndim == 2:
            train_data = np.expand_dims(train_data, axis=1)
        if test_data.ndim == 2:
            test_data = np.expand_dims(test_data, axis=1)

        # Optional resampling to target window length (e.g., 721)
        if getattr(self, 'target_win_len', None) is not None:
            targ = int(self.target_win_len)
            def resample_array(arr, targ_len):
                # arr: (N, C, L)
                N, C, L = arr.shape
                out = np.empty((N, C, targ_len), dtype=np.float32)
                xp = np.arange(L)
                xnew = np.linspace(0, L - 1, targ_len)
                for i in range(N):
                    for c in range(C):
                        try:
                            out[i, c] = np.interp(xnew, xp, arr[i, c])
                        except Exception:
                            out[i, c] = 0.0
                return out

            if train_data.shape[-1] != targ:
                print(f"[MITBIH] Resampling train_data {train_data.shape[-1]} -> {targ}")
                train_data = resample_array(train_data.astype(np.float32, copy=False), targ)
            if test_data.shape[-1] != targ:
                print(f"[MITBIH] Resampling test_data {test_data.shape[-1]} -> {targ}")
                test_data = resample_array(test_data.astype(np.float32, copy=False), targ)

        # Optional per-beat normalization to match MITBIH loader behavior
        if getattr(self, 'per_beat_normalize', True):
            def per_beat_z(x):
                # x: (N, C, L)
                eps = 1e-6
                for i in range(x.shape[0]):
                    for c in range(x.shape[1]):
                        arr = x[i, c]
                        mu = arr.mean()
                        sd = arr.std() + eps
                        x[i, c] = (arr - mu) / sd
                return x
            train_data = per_beat_z(train_data)
            test_data = per_beat_z(test_data)

        # 언더샘플링 + 이피션트 서브샘플링 (train only)
        train_data, train_labels, train_patients, train_feats = \
            self._apply_undersampling(train_data, train_labels, train_patients, train_feats)
        train_data, train_labels, train_patients, train_feats = \
            self._apply_efficiency(train_data, train_labels, train_patients, train_feats)

        self.train_data = train_data
        self.train_labels = train_labels
        self.train_patients = train_patients
        self.train_feats = train_feats

        self.valid_data = test_data
        self.valid_labels = test_labels
        self.valid_patients = test_patients
        self.valid_feats = test_feats

        self.num_train_patients = len(np.unique(train_patients))
        self.num_test_patients = len(np.unique(test_patients))

        print(f"[MITBIH_DANN_DataModule] Loaded:")
        print(f"  Train: {len(train_data)} samples, {self.num_train_patients} patients")
        print(f"  Test : {len(test_data)} samples, {self.num_test_patients} patients")
        print(f"  Feature dim: {self.train_feats.shape[1]}")

    # Loaders -----------------------------------------------------------
    def train_dataloader(self):
        ds = MITBIHDANN_Dataset(
            self.train_data, self.train_labels, self.train_patients, self.train_feats,
            window_type=self.window_type, window_param=self.window_param,
        )
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self._pin_memory,
            drop_last=self._drop_last,
        )

    def val_dataloader(self):
        ds = MITBIHDANN_Dataset(
            self.valid_data, self.valid_labels, self.valid_patients, self.valid_feats,
            window_type=self.window_type, window_param=self.window_param,
        )
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self._pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        return self.val_dataloader()


# -------------------------------------------------------------------------
# Quick test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # ── HiCardi-compatible 인터페이스 사용 예시 ───────────────────────────────
    # split_json: {"base_dir": "/path/to/processed/seg2.0_fs360_MLII_inter"}
    dm = MITBIH_DANN_DataModule(
        split_json="./data/mitbih_split.json",
        return_arg={
            "n_classes": 5,
        },
        undersampling_arg={
            "strategy":     "random",
            "normal_ratio": 1.0,
        },
        efficient_training_arg={
            "variation_factor": 0.5,
        },
        dataloader_arg={
            "batch_size":  64,
            "num_workers": 8,
            "pin_memory":  True,
            "drop_last":   True,
        },
    )

    # ── Legacy 인터페이스도 그대로 동작 ─────────────────────────────────────
    # dm = MITBIH_DANN_DataModule(
    #     base_dir="./data/processed/seg2.0_fs360_MLII_inter",
    #     batch_size=64,
    #     num_workers=8,
    #     window_type="gaussian",
    #     window_param=0.3,
    # )

    dm.setup()

    loader = dm.train_dataloader()
    for x, y, pid, feats in loader:
        print("x:", x.shape)         # [B, 1, L]
        print("y:", y.shape)         # [B]
        print("pid:", pid.shape)     # [B]
        print("feats:", feats.shape) # [B, F]
        break
