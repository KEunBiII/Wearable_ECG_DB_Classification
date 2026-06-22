# -*- coding: utf-8 -*-
from __future__ import annotations

from enum import Enum

import numpy as np


# ── 정규화 ────────────────────────────────────────────────────────────────────

def normalize(x: np.ndarray, mode: str) -> np.ndarray:
    """
    mode = "none"   : 정규화 없이 그대로
    mode = "z"      : (x - mean) / std
    mode = "minmax" : (x - min) / (max - min)
    """
    if mode == "none":
        return x
    if mode == "z":
        mu = x.mean(axis=-1, keepdims=True)
        sd = x.std(axis=-1, keepdims=True) + 1e-6
        return (x - mu) / sd
    if mode == "minmax":
        mn = x.min(axis=-1, keepdims=True)
        mx = x.max(axis=-1, keepdims=True)
        return (x - mn) / (mx - mn + 1e-8)
    raise ValueError(f"normalize는 'none' / 'z' / 'minmax' 중 하나여야 합니다. 받은 값: {mode!r}")


# ── 거리 지표 ─────────────────────────────────────────────────────────────────

class DistanceMetric(str, Enum):
    EUCLIDEAN   = "euclidean"
    COSINE      = "cosine"
    MANHATTAN   = "manhattan"
    CHEBYSHEV   = "chebyshev"
    DTW         = "dtw"          # 소규모 데이터 권장 (O(n²) 비용)
    MAHALANOBIS = "mahalanobis"


# ── 거리 함수 구현 ────────────────────────────────────────────────────────────

def _euclidean(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X - c, axis=-1)


def _cosine(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    dot = X @ c
    nX  = np.linalg.norm(X, axis=-1) + 1e-8
    nc  = np.linalg.norm(c) + 1e-8
    return 1.0 - dot / (nX * nc)


def _manhattan(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.abs(X - c).sum(axis=-1)


def _chebyshev(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.abs(X - c).max(axis=-1)


def _dtw_single(x: np.ndarray, y: np.ndarray) -> float:
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = abs(x[i - 1] - y[j - 1]) + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return float(D[n, m])


def _dtw(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.array([_dtw_single(x, c) for x in X])


def _mahalanobis(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    diff = X - c
    cov  = np.cov(X.T) + np.eye(X.shape[1]) * 1e-6
    VI   = np.linalg.pinv(cov)
    return np.array([np.sqrt(d @ VI @ d) for d in diff])


_DIST_FN = {
    DistanceMetric.EUCLIDEAN:   _euclidean,
    DistanceMetric.COSINE:      _cosine,
    DistanceMetric.MANHATTAN:   _manhattan,
    DistanceMetric.CHEBYSHEV:   _chebyshev,
    DistanceMetric.DTW:         _dtw,
    DistanceMetric.MAHALANOBIS: _mahalanobis,
}


def compute_distances(
    X:        np.ndarray,
    centroid: np.ndarray,
    metric:   "str | DistanceMetric" = DistanceMetric.EUCLIDEAN,
) -> np.ndarray:
    """
    각 행 X[i]와 단일 centroid 벡터 사이의 거리를 계산합니다.

    X        : (N, D)  — 벡터 행렬
    centroid : (D,)    — 기준 벡터
    metric   : DistanceMetric 또는 동일한 값의 문자열
    반환값   : (N,)
    """
    if not isinstance(metric, DistanceMetric):
        metric = DistanceMetric(metric)
    X = np.atleast_2d(X).astype(np.float64)
    c = np.asarray(centroid, dtype=np.float64).ravel()
    return _DIST_FN[metric](X, c)
