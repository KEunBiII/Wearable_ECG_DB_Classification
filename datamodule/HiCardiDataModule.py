# -*- coding: utf-8 -*-
"""
HiCardiDataModule.py  —  HiCardi Holter ECG 데이터 로더 (Clean Architecture)
────────────────────────────────────────────────────────────────────────────────

설계 원칙
─────────
SRP   각 클래스는 단일 책임:
        Config        설정 데이터 + dict 파싱
        BeatMask      정상/비정상 인덱스 캡슐화
        Masker        mask 계산
        IUndersampler 정상 풀에서 유지 인덱스 선택 (Stage 1 전략)
        IPrototyper   풀에서 n_keep개 선택        (Stage 2 전략)
        Stage         한 파이프라인 단계 실행
        DataPipeline  여러 stage 오케스트레이션
        LoaderFactory DataLoader 생성
        DataModule    Lightning 통합 + public API

OCP   새 샘플링/프로토타이핑 전략 추가 시 Registry에 등록만 → 기존 코드 무변경

DIP   DataModule이 구체 전략이 아닌 인터페이스(IUndersampler, IPrototyper)에 의존

상태  모든 stage 결과를 StageResult로 캡슐화하여 PipelineResult에 누적 저장.
      → balance_summary 등 분석 API는 추정이 아닌 실측 데이터를 읽음.

파이프라인
─────────
  raw_dataset
    → UndersamplingStage  (정상/비정상 비율 조정)
    → EfficiencyStage     (stratified prototyping subsampling — 비율 보존)
    → IndexSubset (active)
    → DataLoader          (hicardi_collate_fn + optional Mixup)

사용 예시
─────────
  dm = HiCardiDataModule(
      split_json="split.json",
      return_arg={"waveform": True, "labels": True, "mode": "beat", "n_classes": 7},
      undersampling_arg={"strategy": "centroid_near", "normal_ratio": 2.0},
      efficient_training_arg={
          "apply_threshold": 200_000, "variation_factor": 0.3,
          "prototyping": "random",
          "mixup": {"alpha": 0.2, "apply_prob": 0.5},
      },
      dataloader_arg={"batch_size": 128, "num_workers": 8},
  )
  dm.setup()
  dm.print_balance_summary()

  # 런타임 재설정 (setup 재호출 불필요)
  dm.resample("train", efficient_training_arg={"variation_factor": 0.5,
                                               "prototyping": "hard_mining"})
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# utils 폴더 내의 dataModuleUtils에서 필요한 클래스 및 상수들을 임포트
# datamodule/HiCardiDataModule.py 상단 수정
from .utils.dataModuleUtils import (
    DatasetMode, ReturnConfig, DatasetConfig, UndersamplingConfig,
    EfficiencyConfig, DataLoaderConfig, BeatLabelMasker, LoaderFactory,
    DataPipeline, HiCardiDataset, HiCardiSequenceDataset, IndexSubset,
    RecordCentroidAnalyzer, BeatMask, StageResult, PipelineResult,
    UndersamplingStage, EfficiencyStage,               
    SamplingConfig, SamplingStrategy, BalancedSubset,   
    DEFAULT_BEAT_CLASSES, HICARDI_LABEL_COL_MAP,        
)
from .utils.io_utils import load_json, save_json, hicardi_collate_fn
from .utils.math_utils import DistanceMetric
from .utils.analyzer import PopulationAnalyzer

try:
    import lightning as L
    _LDM_BASE = L.LightningDataModule
except ImportError:
    L = None
    _LDM_BASE = object



class HiCardiDataModule(_LDM_BASE):  # type: ignore[misc]
    """HiCardi Holter ECG Lightning DataModule (4-dict 설정).

    Public API는 모두 PipelineResult를 읽어서 응답하므로 상태가 항상 일관됨.
    setup과 resample은 같은 DataPipeline.run()을 호출 → 단계 우회 버그 불가.
    """

    def __init__(
        self,
        split_json:             str,
        return_arg:             Mapping[str, Any] = None,
        undersampling_arg:      Mapping[str, Any] = None,
        efficient_training_arg: Mapping[str, Any] = None,
        dataloader_arg:         Mapping[str, Any] = None,
    ):
        super().__init__()
        self.split_json = split_json

        # 1) 설정 파싱 (각 dataclass.from_dict 위임)
        self.return_cfg  = ReturnConfig.from_dict(return_arg or {})
        self.dataset_cfg = DatasetConfig.from_dict(return_arg or {})
        self.und_cfg     = UndersamplingConfig.from_dict(undersampling_arg or {})
        self.eff_cfg     = EfficiencyConfig.from_dict(efficient_training_arg or {})
        self.loader_cfg  = DataLoaderConfig.from_dict(dataloader_arg or {})

        # 2) 컴포넌트 조립 (DI)
        self.masker = BeatLabelMasker(exclude_normal_col=self.und_cfg.exclude_normal_col)
        self.loader_factory = LoaderFactory(
            self.loader_cfg, self.eff_cfg.mixup, mixup_seed=self.eff_cfg.seed,
        )

        # 3) 상태
        self._pipelines: Dict[str, Optional[PipelineResult]] = {}
        self._split_data: Dict[str, list] = {}

    # ── setup ────────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        if self._pipelines:   # 이미 setup 완료 → 재실행 방지 (trainer.fit 내부 호출 중복 방지)
            return
        self._split_data = load_json(self.split_json)
        self._normalize_cache_paths(self._split_data, self.dataset_cfg.cache_root)

        for split, recs in self._split_data.items():
            if not recs:
                self._pipelines[split] = None
                continue
            print(f"  [{split}] 파이프라인 구성 중 ({len(recs)} 레코드)...")
            raw = self._build_raw_dataset(recs)
            self._pipelines[split] = self._build_pipeline(split).run(raw) \
                if self.dataset_cfg.mode == DatasetMode.BEAT \
                else self._sequence_passthrough(raw)
            print(f"  [{split}] 완료")

    def _build_pipeline(self, split: str) -> DataPipeline:
        und = (UndersamplingStage(self.und_cfg, self.masker)
               if self.und_cfg.applies_to(split) else None)
        eff = (EfficiencyStage(self.eff_cfg, self.und_cfg.distance_metric)
               if self.eff_cfg.applies_to(split) else None)
        return DataPipeline(self.masker, und, eff)

    @staticmethod
    def _sequence_passthrough(raw: Dataset) -> PipelineResult:
        """sequence 모드: mask/undersampling 의미 없음 → raw 그대로."""
        empty = BeatMask()
        pt = StageResult(indices=[], mask=empty, applied=False)
        return PipelineResult(
            raw_dataset=raw, initial_mask=empty,
            stage1_result=pt, stage2_result=pt,
            active_dataset=raw,
        )

    @staticmethod
    def _normalize_cache_paths(split_data: Dict[str, list], cache_root: Optional[str]) -> None:
        if cache_root is None:
            return
        root = str(Path(cache_root).resolve())
        for recs in split_data.values():
            for rec in recs:
                old = rec.get("cache_path", "")
                if old:
                    parts = Path(old).parts
                    rec["cache_path"] = os.path.join(root, *parts[-2:])

    def _build_raw_dataset(self, records: list) -> Dataset:
        common = dict(
            records       = records,
            return_cfg    = self.return_cfg,
            normalize     = self.dataset_cfg.normalize,
            label_names   = self.dataset_cfg.label_names,
            n_classes     = self.dataset_cfg.n_classes,
            cohort_filter = self.dataset_cfg.cohort_filter,
            min_beats     = self.dataset_cfg.min_beats,
        )
        if self.dataset_cfg.mode == DatasetMode.BEAT:
            return HiCardiDataset(**common, window_half=self.dataset_cfg.window_half)
        return HiCardiSequenceDataset(
            **common,
            seq_len=self.dataset_cfg.seq_len,
            stride =self.dataset_cfg.stride,
        )

    # ── 런타임 재실행 (setup 재호출 불필요) ─────────────────────────────────

    def resample(
        self,
        split: str = "train",
        undersampling_arg:      Optional[Mapping[str, Any]] = None,
        efficient_training_arg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """split의 파이프라인을 새 설정으로 재실행.
        둘 다 None이면 현재 설정으로 다시 돌림 (시드 변경 등 부가 효과)."""
        if undersampling_arg is not None:
            self.und_cfg = UndersamplingConfig.from_dict(undersampling_arg)
            self.masker  = BeatLabelMasker(exclude_normal_col=self.und_cfg.exclude_normal_col)
        if efficient_training_arg is not None:
            self.eff_cfg = EfficiencyConfig.from_dict(efficient_training_arg)
            self.loader_factory = LoaderFactory(
                self.loader_cfg, self.eff_cfg.mixup, mixup_seed=self.eff_cfg.seed,
            )

        p = self._pipelines.get(split)
        if p is None or self.dataset_cfg.mode != DatasetMode.BEAT:
            return
        self._pipelines[split] = self._build_pipeline(split).run(p.raw_dataset)

    # ── Dataset 접근 ────────────────────────────────────────────────────────

    def get_pipeline(self, split: str) -> Optional[PipelineResult]:
        return self._pipelines.get(split)

    def get_dataset(self, split: str) -> Optional[Dataset]:
        p = self._pipelines.get(split)
        return p.active_dataset if p else None

    def get_raw_dataset(self, split: str) -> Optional[Dataset]:
        p = self._pipelines.get(split)
        return p.raw_dataset if p else None

    def get_base_dataset(self, split: str) -> Optional[Dataset]:
        """하위호환 alias."""
        return self.get_raw_dataset(split)

    # ── 인덱스 접근 ─────────────────────────────────────────────────────────

    def get_normal_indices(self, split: str) -> List[int]:
        p = self._pipelines.get(split)
        return list(p.initial_mask.normal) if p else []

    def get_abnormal_indices(self, split: str) -> List[int]:
        p = self._pipelines.get(split)
        return list(p.initial_mask.abnormal) if p else []

    def get_active_indices(self, split: str) -> List[int]:
        p = self._pipelines.get(split)
        return list(p.stage2_result.indices) if p else []

    # ── DataLoader ──────────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        return self.loader_factory.train_loader(self.get_dataset("train"))

    def val_dataloader(self) -> DataLoader:
        return self.loader_factory.eval_loader(self.get_dataset("val"))

    def test_dataloader(self) -> DataLoader:
        return self.loader_factory.eval_loader(self.get_dataset("test"))

    def bce_dataloader(
        self,
        split:       str,
        batch_size:  int,
        num_workers: int,
        pin_memory:  bool = True,
    ) -> Optional[DataLoader]:
        """active_dataset → BCEWithLogitsLoss 호환 (waveform, y_multihot) DataLoader.

        sentinel label(-1) 샘플을 배치 단위로 제거하고
        multi-hot float 레이블을 그대로 반환합니다.
        """
        ds = self.get_dataset(split)
        if ds is None or len(ds) == 0:
            return None

        def _collate(batch):
            b     = hicardi_collate_fn(batch)
            X     = b["waveform"]                       # (B, win_len)
            y_raw = b["labels"]                         # (B, n_classes) multi-hot
            valid = y_raw.min(dim=1).values >= 0        # sentinel -1 제거
            return X[valid], y_raw[valid].float()

        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = (split == "train"),
            collate_fn  = _collate,
        )

    def extract_numpy(
        self,
        split:       str = "train",
        max_samples: int = 5_000,
        seed:        int = 42,
        num_workers: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """시각화 전용 소규모 numpy 서브샘플 (X, y_argmax).

        max_samples 개만 꺼내므로 학습 루프에는 사용하지 않습니다.
        학습에는 bce_dataloader()를 사용하세요.
        argmax는 분포 집계 및 임베딩 시각화 목적에 한합니다.
        """
        from torch.utils.data import Subset
        ds = self.get_dataset(split)
        if ds is None or len(ds) == 0:
            return np.zeros((0, 1), np.float32), np.zeros(0, np.int64)
        rng  = np.random.default_rng(seed)
        idxs = rng.choice(len(ds), min(len(ds), max_samples), replace=False)
        subset = Subset(ds, idxs.tolist())
        loader = DataLoader(
            subset,
            batch_size=256,
            num_workers=num_workers,
            pin_memory=False,
            shuffle=False,
        )
        waves, labels = [], []
        n_batches = len(loader)
        print(f"  extract_numpy [{split}]: 0/{n_batches} batches", end="", flush=True)
        for i, batch in enumerate(loader, 1):
            lbl  = batch["labels"].numpy()           # (B, n_classes)
            valid = lbl.min(axis=1) >= 0             # sentinel 제거
            if valid.any():
                waves.append(batch["waveform"][valid].numpy())
                labels.append(lbl[valid].argmax(axis=1))
            print(f"\r  extract_numpy [{split}]: {i}/{n_batches} batches", end="", flush=True)
        print()
        if not waves:
            return np.zeros((0, 1), np.float32), np.zeros(0, np.int64)
        return np.concatenate(waves).astype(np.float32), np.concatenate(labels).astype(np.int64)

    # ── 분석 (실측 카운트 — 추정 없음) ──────────────────────────────────────

    def balance_summary(self, split: str = "train") -> Dict[str, Any]:
        p = self._pipelines.get(split)
        if p is None:
            return {}
        b, s1, s2 = p.initial_mask, p.stage1_result.mask, p.stage2_result.mask
        return {
            "split":              split,
            "und_strategy":       self.und_cfg.strategy.value,
            "und_applied":        p.stage1_result.applied,
            "eff_prototyping":    self.eff_cfg.prototyping.value,
            "eff_applied":        p.stage2_result.applied,
            "before":              self._snap(b),
            "after_undersampling": self._snap(s1),
            "after_efficiency":    self._snap(s2),
        }

    @staticmethod
    def _snap(m: BeatMask) -> Dict[str, Any]:
        return {"normal":   m.n_normal,
                "abnormal": m.n_abnormal,
                "total":    m.n_total,
                "ratio":    round(m.ratio, 3)}

    def print_balance_summary(self) -> None:
        print("\n파이프라인 요약")
        print("=" * 80)
        for split in ("train", "val", "test"):
            s = self.balance_summary(split)
            if not s:
                continue
            print(f"\n[{split}]  und={s['und_strategy']} (applied={s['und_applied']})"
                  f"   eff={s['eff_prototyping']} (applied={s['eff_applied']})")
            for key, label in [
                ("before",              "원본       "),
                ("after_undersampling", "Stage 1 후 "),
                ("after_efficiency",    "Stage 2 후 "),
            ]:
                d = s[key]
                print(f"  {label}— 정상: {d['normal']:>8,}  비정상: {d['abnormal']:>8,}"
                      f"  합계: {d['total']:>8,}  비율: {d['ratio']:>6.2f}")

    # ── 센트로이드 분석 위임 ────────────────────────────────────────────────

    def centroid_analyzer(self, split: str = "train") -> RecordCentroidAnalyzer:
        ds = self.get_raw_dataset(split)
        if not isinstance(ds, HiCardiDataset):
            raise ValueError(f"split '{split}'가 비어 있거나 beat 모드가 아닙니다.")
        return RecordCentroidAnalyzer(ds)

    def distance_stats(self, split="train", level="record", metric=DistanceMetric.EUCLIDEAN):
        return self.centroid_analyzer(split).distance_stats(level=level, metric=metric)

    def print_distance_stats(self, split="train", level="record", metric=DistanceMetric.EUCLIDEAN):
        self.centroid_analyzer(split).print_stats(level=level, metric=metric)

    def export_centroids(self, save_dir, split="train", level="record"):
        return self.centroid_analyzer(split).export_centroids(save_dir, level)

    def population_analyzer(self, splits: Optional[List[str]] = None) -> PopulationAnalyzer:
        recs = []
        for key, lst in self._split_data.items():
            if splits is None or key in splits:
                recs.extend(lst)
        return PopulationAnalyzer(recs)

    def dataset_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        for split in ("train", "val", "test"):
            p = self._pipelines.get(split)
            if p is None:
                info[split] = None
                continue
            info[split] = {
                "n_items_raw":    len(p.raw_dataset),
                "n_items_active": len(p.active_dataset),
                "n_records":      getattr(p.active_dataset, "n_records", "?"),
                "cohorts":        getattr(p.active_dataset, "cohorts", []),
                "n_subjects":     len(getattr(p.active_dataset, "subjects", [])),
                "stage1_applied": p.stage1_result.applied,
                "stage2_applied": p.stage2_result.applied,
            }
        return info

    # ── 하위 호환 properties ────────────────────────────────────────────────

    @property
    def ds_train(self) -> Optional[Dataset]: return self.get_dataset("train")
    @property
    def ds_val(self)   -> Optional[Dataset]: return self.get_dataset("val")
    @property
    def ds_test(self)  -> Optional[Dataset]: return self.get_dataset("test")
    @property
    def mode(self) -> str:                   return self.dataset_cfg.mode.value
    @property
    def label_names(self) -> List[str]:      return self.dataset_cfg.label_names
    @property
    def n_classes(self) -> int:              return self.dataset_cfg.n_classes
    @property
    def batch_size(self) -> int:             return self.loader_cfg.batch_size
    @property
    def num_workers(self) -> int:            return self.loader_cfg.num_workers
    @property
    def sampling_cfg(self) -> UndersamplingConfig: return self.und_cfg
    @property
    def efficiency_cfg(self) -> EfficiencyConfig:  return self.eff_cfg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸 함수 (하위 호환)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_split_json(path: str) -> Dict[str, list]:
    return load_json(path)


def save_split_json(split: Dict[str, list], path: str) -> None:
    save_json(split, path)