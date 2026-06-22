# -*- coding: utf-8 -*-
"""
plot_beats.py — beat cache(X.npy / Y.npy)에서 클래스별 비트 샘플을 시각화

사용법:
    python plot_beats.py                        # 기본 캐시 경로 사용
    python plot_beats.py --cache <cache_dir>    # 캐시 루트 지정
    python plot_beats.py --record <record_dir>  # 레코드 1개 직접 지정
    python plot_beats.py --save result.png      # 파일로 저장 (화면 표시 안 함)
"""

import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── HiCardi 라벨 컬럼 매핑 ────────────────────────────────────────────────────
HICARDI_LABEL_COL_MAP: dict[str, int] = {
    "Normal":      0,
    "VF/VT":       2,
    "VPC":         3,
    "Bigeminy":    5,
    "Trigeminy":   6,
    "Bradycardia": 8,
    "AF/AFL":      12,
    "APC":         14,
    "Sinus Tachy": 16,
}
COL_TO_NAME = {v: k for k, v in HICARDI_LABEL_COL_MAP.items()}

LABEL_COLORS = {
    "Normal":      "#44cc66",
    "VF/VT":       "#cc2222",
    "VPC":         "#ff5555",
    "Bigeminy":    "#ff88cc",
    "Trigeminy":   "#ffaa88",
    "Bradycardia": "#4488ff",
    "AF/AFL":      "#aa44ff",
    "APC":         "#ff9900",
    "Sinus Tachy": "#ffdd00",
    "Unknown":     "#888888",
}


# =============================================================================
# 데이터 로딩
# =============================================================================

def load_record(record_dir: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    p = Path(record_dir)
    X  = np.load(p / "X.npy",  mmap_mode="r")   # (B, 501)
    Y  = np.load(p / "Y.npy",  mmap_mode="r")   # (B, 27)
    fs = float(np.load(p / "fs.npy")[0])
    return X, Y, fs


def collect_all_records(cache_root: str | Path) -> list[Path]:
    """캐시 루트 아래 X.npy 를 가진 레코드 폴더를 재귀 탐색."""
    root = Path(cache_root)
    return sorted(p.parent for p in root.rglob("X.npy"))


def build_class_index(
    records: list[Path],
) -> dict[str, list[tuple[Path, int]]]:
    """
    각 클래스별로 (record_dir, beat_idx) 목록을 수집.
    multi-hot 비트는 priority 순서로 첫 번째 활성 클래스에 귀속.
    """
    priority = list(HICARDI_LABEL_COL_MAP.keys())   # Normal 우선
    class_index: dict[str, list] = {k: [] for k in priority}
    class_index["Unknown"] = []

    for rec_dir in records:
        try:
            _, Y, _ = load_record(rec_dir)
        except Exception as e:
            print(f"[SKIP] {rec_dir.name}: {e}")
            continue

        for i in range(len(Y)):
            assigned = False
            for name in priority:
                col = HICARDI_LABEL_COL_MAP[name]
                if col < Y.shape[1] and Y[i, col] > 0:
                    class_index[name].append((rec_dir, i))
                    assigned = True
                    break
            if not assigned:
                class_index["Unknown"].append((rec_dir, i))

    return class_index


# =============================================================================
# 플롯
# =============================================================================

def plot_class_samples(
    class_index: dict[str, list],
    n_samples: int = 8,
    fs: float = 250.0,
    save_path: str | None = None,
):
    """클래스별로 n_samples개 비트를 격자 플롯."""
    active_classes = [k for k, v in class_index.items() if len(v) > 0]
    if not active_classes:
        print("[ERROR] 플롯할 클래스가 없습니다.")
        return

    n_cls  = len(active_classes)
    n_cols = n_samples
    n_rows = n_cls

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.6, n_rows * 1.8),
                             squeeze=False)
    fig.suptitle("HiCardi — Class Beat Samples", fontsize=13, y=1.01)

    t = np.arange(501) / fs * 1000   # ms

    for r, cls_name in enumerate(active_classes):
        samples = class_index[cls_name]
        color   = LABEL_COLORS.get(cls_name, "#888888")
        n_avail = len(samples)

        # 라벨 (왼쪽)
        axes[r, 0].set_ylabel(f"{cls_name}\n(n={n_avail:,})",
                               fontsize=8, rotation=0,
                               labelpad=60, va="center")

        rng = np.random.default_rng(42)
        chosen = rng.choice(n_avail, size=min(n_samples, n_avail), replace=False)

        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(chosen):
                rec_dir, beat_idx = samples[chosen[c]]
                X, _, _ = load_record(rec_dir)
                x = np.array(X[beat_idx], dtype=np.float32)
                ax.plot(t, x, linewidth=0.8, color=color)
                ax.axvline(x=250 / fs * 1000, color="#999", linewidth=0.5,
                           linestyle="--", alpha=0.6)   # R-peak 중심
            else:
                ax.set_visible(False)
                continue

            ax.set_xlim(t[0], t[-1])
            ax.tick_params(labelsize=5, length=2, pad=1)
            ax.spines[["top", "right"]].set_visible(False)
            if r < n_rows - 1:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_xlabel("")
            if r == n_rows - 1:
                ax.set_xlabel("ms", fontsize=6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()


def plot_label_distribution(
    class_index: dict[str, list],
    save_path: str | None = None,
):
    """클래스별 비트 수 막대 그래프."""
    names  = [k for k in class_index if len(class_index[k]) > 0]
    counts = [len(class_index[k]) for k in names]
    colors = [LABEL_COLORS.get(k, "#888") for k in names]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("HiCardi — Beat Label Distribution", fontsize=12)
    ax.set_ylabel("Beat count")
    ax.set_xlabel("Class")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                f"{cnt:,}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()

    if save_path:
        dist_path = str(save_path).replace(".png", "_dist.png")
        plt.savefig(dist_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {dist_path}")
    else:
        plt.show()


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="HiCardi beat cache 시각화")
    ap.add_argument("--cache",   type=str, default=None,
                    help="beat cache 루트 (X.npy 가 있는 폴더들의 부모)")
    ap.add_argument("--record",  type=str, default=None,
                    help="레코드 폴더 1개 직접 지정 (--cache 대신)")
    ap.add_argument("--n",       type=int, default=8,
                    help="클래스당 샘플 수 (기본 8)")
    ap.add_argument("--save",    type=str, default=None,
                    help="저장 경로 (.png). 없으면 화면 출력")
    args = ap.parse_args()

    # ── 레코드 수집 ─────────────────────────────────────────────────────────
    if args.record:
        records = [Path(args.record)]
    elif args.cache:
        records = collect_all_records(args.cache)
    else:
        # 기본값: Downloads의 추출된 캐시
        default_cache = Path.home() / "Downloads" / \
            "hicardi_beat_cache-20260607T084333Z-3-001" / \
            "hicardi_beat_cache" / "hicardi_beat_cache"
        if default_cache.exists():
            records = collect_all_records(default_cache)
        else:
            print("[ERROR] --cache 또는 --record 경로를 지정하세요.")
            return

    print(f"[Info] 레코드 {len(records)}개 로드")

    # ── 클래스 인덱스 구축 ───────────────────────────────────────────────────
    class_index = build_class_index(records)
    for k, v in class_index.items():
        if v:
            print(f"  {k:12s}: {len(v):,} beats")

    # ── 플롯 ─────────────────────────────────────────────────────────────────
    if args.save:
        matplotlib.use("Agg")

    plot_label_distribution(class_index, save_path=args.save)
    plot_class_samples(class_index, n_samples=args.n, save_path=args.save)


if __name__ == "__main__":
    main()
