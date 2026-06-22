# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from typing import Any, Dict, List, Optional

import numpy as np

from .dataset import MemmapRecordStore
from .math_utils import compute_distances


class PopulationAnalyzer:
    """
    split.json 또는 index.csv의 레코드 목록을 받아 인구·병원 통계를 제공합니다.

    사용 예
    ───────
    analyzer = PopulationAnalyzer.from_split_json("split.json")
    analyzer.print_summary()
    """

    def __init__(self, records: List[Dict[str, str]]):
        self.records = records

    @classmethod
    def from_split_json(cls, path: str, splits: Optional[List[str]] = None) -> "PopulationAnalyzer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        recs = []
        for key, lst in data.items():
            if splits is None or key in splits:
                recs.extend(lst)
        return cls(recs)

    @classmethod
    def from_index_csv(cls, path: str) -> "PopulationAnalyzer":
        with open(path, "r", encoding="utf-8") as f:
            return cls([dict(row) for row in csv.DictReader(f)])

    def cohort_summary(self) -> Dict[str, Dict[str, int]]:
        summary: Dict[str, Any] = {}
        for r in self.records:
            c = r.get("cohort", "unknown")
            if c not in summary:
                summary[c] = {"n_records": 0, "n_subjects": set(), "n_beats": 0}
            summary[c]["n_records"] += 1
            summary[c]["n_subjects"].add(r.get("subject_id", ""))
            summary[c]["n_beats"]   += int(r.get("num_beats", 0))
        for c in summary:
            summary[c]["n_subjects"] = len(summary[c]["n_subjects"])
        return summary

    def subject_beat_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.records:
            sid = r.get("subject_id", "unknown")
            counts[sid] = counts.get(sid, 0) + int(r.get("num_beats", 0))
        return counts

    def label_distribution(
        self,
        label_names: Optional[List[str]] = None,
        per_cohort: bool = False,
    ) -> Dict[str, Any]:
        store = MemmapRecordStore()
        total: Dict[str, int] = {}
        by_cohort: Dict[str, Dict[str, int]] = {}

        for r in self.records:
            cp = r.get("cache_path", "")
            if not cp:
                continue
            try:
                rec = store.get(cp)
            except Exception:
                continue
            if "labels" not in rec:
                continue

            Y      = np.asarray(rec["labels"])
            cohort = r.get("cohort", "unknown")
            by_cohort.setdefault(cohort, {})

            if Y.ndim == 1:
                for v in Y.tolist():
                    k = label_names[int(v)] if label_names and int(v) < len(label_names) else str(int(v))
                    total[k]             = total.get(k, 0) + 1
                    by_cohort[cohort][k] = by_cohort[cohort].get(k, 0) + 1
            else:
                for ci in range(Y.shape[1]):
                    k   = label_names[ci] if label_names and ci < len(label_names) else f"cls_{ci}"
                    cnt = int(Y[:, ci].sum())
                    total[k]             = total.get(k, 0) + cnt
                    by_cohort[cohort][k] = by_cohort[cohort].get(k, 0) + cnt

        return {"total": total, "per_cohort": by_cohort} if per_cohort else {"total": total}

    def demographic_summary(self) -> Dict[str, Dict[str, int]]:
        store = MemmapRecordStore()
        agg: Dict[str, Dict[str, int]] = {}
        for r in self.records:
            cp = r.get("cache_path", "")
            if not cp:
                continue
            try:
                rec = store.get(cp)
            except Exception:
                continue
            for key, val in rec.get("demo", {}).items():
                agg.setdefault(key, {})
                v = str(val)
                agg[key][v] = agg[key].get(v, 0) + 1
        return agg

    def print_summary(self, label_names: Optional[List[str]] = None) -> None:
        subj_beats  = self.subject_beat_counts()
        total_beats = sum(subj_beats.values())

        print("=" * 60)
        print("HiCardi Population Summary")
        print("=" * 60)
        print(f"총 레코드  : {len(self.records):,}")
        print(f"총 피험자  : {len(subj_beats):,}")
        print(f"총 박동 수 : {total_beats:,}")

        print("\n─ 코호트별 ─")
        for cohort, info in sorted(self.cohort_summary().items()):
            pct = info["n_beats"] / max(total_beats, 1) * 100
            print(f"  {cohort:<32s}  recs={info['n_records']:4d}  "
                  f"subj={info['n_subjects']:4d}  "
                  f"beats={info['n_beats']:8,}  ({pct:.1f}%)")

        lbl_dist = self.label_distribution(label_names=label_names)["total"]
        if lbl_dist:
            print("\n─ 레이블 분포 ─")
            for k, v in sorted(lbl_dist.items(), key=lambda x: -x[1]):
                print(f"  {k:<20s}: {v:8,}")

        demo = self.demographic_summary()
        if demo:
            print("\n─ 인구통계 ─")
            for fname, counts in demo.items():
                print(f"  {fname}:")
                for val, cnt in sorted(counts.items()):
                    print(f"    {val:<15s}: {cnt:,}")

        print("=" * 60)


class CentroidAnalyzer:
    """
    beat feature 벡터들의 클래스별 centroid를 계산하고,
    임의 샘플과의 거리를 기반으로 분석합니다.

    사용 예
    ───────
    ca = CentroidAnalyzer(features, labels, label_names=["Normal", "VPC", ...])
    ca.fit()
    dists = ca.distances_to_centroids(query_features)  # (N, n_cls)
    nearest = ca.nearest_class(query_features)          # (N,)
    """

    def __init__(
        self,
        features:    np.ndarray,
        labels:      np.ndarray,
        label_names: Optional[List[str]] = None,
        metric:      str                 = "euclidean",
    ):
        """
        features : (N, D)  float  — beat feature 벡터
        labels   : (N,)    int    — 클래스 인덱스 (multi-hot 불가)
        """
        self.features    = np.asarray(features, dtype=np.float64)
        self.labels      = np.asarray(labels,   dtype=np.int64)
        self.label_names = label_names
        self.metric      = metric
        self.centroids_: Optional[np.ndarray] = None   # (n_cls, D)
        self.classes_:   Optional[np.ndarray] = None   # (n_cls,)

    def fit(self) -> "CentroidAnalyzer":
        """클래스별 평균 벡터(centroid)를 계산합니다."""
        classes = np.unique(self.labels)
        centroids = np.stack(
            [self.features[self.labels == c].mean(axis=0) for c in classes]
        )
        self.classes_   = classes
        self.centroids_ = centroids
        return self

    def distances_to_centroids(self, queries: np.ndarray) -> np.ndarray:
        """
        queries  : (N, D)
        반환값   : (N, n_cls)  — 각 쿼리와 모든 centroid 사이의 거리
        """
        if self.centroids_ is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        queries = np.atleast_2d(queries).astype(np.float64)
        return np.stack(
            [compute_distances(queries, c, metric=self.metric) for c in self.centroids_],
            axis=1,
        )

    def nearest_class(self, queries: np.ndarray) -> np.ndarray:
        """
        queries  : (N, D)
        반환값   : (N,) int  — 가장 가까운 centroid의 클래스 인덱스
        """
        dists = self.distances_to_centroids(queries)   # (N, n_cls)
        idx   = dists.argmin(axis=1)
        return self.classes_[idx]

    def intra_class_variance(self) -> Dict[int, float]:
        """클래스별 intra-class variance (평균 제곱 거리)를 반환합니다."""
        if self.centroids_ is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        result: Dict[int, float] = {}
        for i, c in enumerate(self.classes_):
            feats = self.features[self.labels == c]
            dists = compute_distances(feats, self.centroids_[i], metric=self.metric)
            result[int(c)] = float(dists.mean())
        return result
