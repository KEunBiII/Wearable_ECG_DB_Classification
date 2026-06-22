# -*- coding: utf-8 -*-
"""
DataEfficientTrainer — 데이터 효율적 학습 + 시각화 통합 클래스

학습(fit/test), 임베딩 분석(PCA·t-SNE·UMAP), 센트로이드 거리,
학습 곡선·혼동 행렬·최종 성능 비교까지 한 곳에서 처리합니다.

사용 예
───────
from models.SimpleCNN1D import SimpleCNN1D
from trainer.DataEfficientTrainer import DataEfficientTrainer

model   = SimpleCNN1D(win_len=501, n_classes=7)
trainer = DataEfficientTrainer(model, n_classes=7, lr=1e-3)
history = trainer.fit(train_loader, val_loader, n_epochs=20)
metrics = trainer.test(test_loader)

# 임베딩 시각화 (data-level)
DataEfficientTrainer.plot_embeddings(X, y, label_names, title="train", out_path="embed.png")

# 학습 결과 시각화 (training-level)
trainer.plot_learning_curves({"Random": hist_r, "Near": hist_n}, out_path="curves.png")
trainer.plot_final_metrics(results, out_path="metrics.png")
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
)

try:
    import umap as umap_lib; HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from datamodule.utils.math_utils import DistanceMetric, compute_distances
except ImportError:
    try:
        from utils.math_utils import DistanceMetric, compute_distances
    except ImportError:
        DistanceMetric = None
        compute_distances = None

warnings.filterwarnings("ignore")


class DataEfficientTrainer:
    """
    Parameters
    ──────────
    model        nn.Module 분류기 (SimpleCNN1D 또는 임의 모델)
    n_classes    클래스 수
    lr           AdamW 학습률
    weight_decay AdamW 가중치 감쇠
    threshold    sigmoid 출력 이진화 임계값 (기본 0.5)
    device       None이면 CUDA 자동 감지
    criterion    None이면 BCEWithLogitsLoss
    label_names  시각화용 클래스 이름 목록
    """

    def __init__(
        self,
        model:        nn.Module,
        n_classes:    int   = 7,
        lr:           float = 1e-3,
        weight_decay: float = 1e-2,
        threshold:    float = 0.5,
        device:       Optional[torch.device] = None,
        criterion:    Optional[nn.Module]    = None,
        label_names:  Optional[List[str]]    = None,
    ):
        self.device      = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model       = model.to(self.device)
        self.n_classes   = n_classes
        self.threshold   = threshold
        self.criterion   = criterion or nn.BCEWithLogitsLoss()
        self.label_names = label_names or [str(i) for i in range(n_classes)]
        self.optimizer   = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self._best_state: Optional[dict] = None
        self.best_val_f1: float = 0.0
        self._colors = plt.cm.get_cmap("tab10", n_classes).colors

    # ══════════════════════════════════════════════════════════════════════
    # 학습 메서드
    # ══════════════════════════════════════════════════════════════════════

    def train_epoch(self, loader: DataLoader, ep: int = 0, n_epochs: int = 0) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, n = 0.0, 0, 0
        n_batches = len(loader)
        for i, (X_b, y_b) in enumerate(loader, 1):
            X_b = X_b.to(self.device)
            y_b = y_b.to(self.device).float()
            self.optimizer.zero_grad()
            logits = self.model(X_b)
            loss   = self.criterion(logits, y_b)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(y_b)
            preds   = (torch.sigmoid(logits) > self.threshold).float()
            correct += (preds == y_b).all(dim=1).sum().item()  # exact match
            n       += len(y_b)
            if i % max(1, n_batches // 20) == 0 or i == n_batches:
                print(f"\r  [train ep {ep}/{n_epochs}] {i}/{n_batches} batches  loss={total_loss/max(n,1):.4f}", end="", flush=True)
        print()
        return {"loss": total_loss / max(n, 1), "acc": correct / max(n, 1)}

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader, label: str = "val") -> Dict[str, Any]:
        self.model.eval()
        total_loss, n = 0.0, 0
        all_pred, all_true = [], []
        n_batches = len(loader)
        for i, (X_b, y_b) in enumerate(loader, 1):
            X_b = X_b.to(self.device)
            y_b = y_b.to(self.device).float()
            logits = self.model(X_b)
            total_loss += self.criterion(logits, y_b).item() * len(y_b)
            preds = (torch.sigmoid(logits) > self.threshold).cpu().numpy().astype(np.int32)
            all_pred.append(preds)
            all_true.append(y_b.cpu().numpy().astype(np.int32))
            n += len(y_b)
            if i % max(1, n_batches // 20) == 0 or i == n_batches:
                print(f"\r  [{label}] {i}/{n_batches} batches", end="", flush=True)
        print()
        y_pred = np.vstack(all_pred)   # (N, n_classes) multi-hot int
        y_true = np.vstack(all_true)   # (N, n_classes) multi-hot int
        return {
            "loss":     total_loss / max(n, 1),
            "acc":      accuracy_score(y_true, y_pred),          # exact match
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "y_pred":   y_pred,
            "y_true":   y_true,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        n_epochs:     int,
        scheduler:    Optional[Any] = None,
        verbose:      bool = False,
    ) -> Dict[str, List[float]]:
        """학습 후 history dict 반환. best val F1 가중치를 내부 보존."""
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs
            )
        history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [], "val_f1": [],
        }
        self.best_val_f1 = 0.0
        self._best_state  = None

        for ep in range(1, n_epochs + 1):
            tr  = self.train_epoch(train_loader, ep=ep, n_epochs=n_epochs)
            val = self.eval_epoch(val_loader, label="val")
            scheduler.step()
            for key, val_key in [("train_loss", "loss"), ("train_acc", "acc")]:
                history[key].append(tr[val_key])
            for key, val_key in [("val_loss", "loss"), ("val_acc", "acc"), ("val_f1", "macro_f1")]:
                history[key].append(val[val_key])
            if val["macro_f1"] >= self.best_val_f1:
                self.best_val_f1 = val["macro_f1"]
                self._best_state  = copy.deepcopy(self.model.state_dict())
            print(f"  ep {ep:3d}/{n_epochs}  loss={tr['loss']:.4f}  val_f1={val['macro_f1']:.4f}")

        self._load_best()
        return history

    def test(self, loader: DataLoader) -> Dict[str, Any]:
        """best 가중치로 테스트 세트 평가."""
        self._load_best()
        return self.eval_epoch(loader, label="test")

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def _load_best(self) -> None:
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

    # ══════════════════════════════════════════════════════════════════════
    # 데이터 분포 시각화 (staticmethod — 모델·학습 상태 불필요)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def plot_class_distribution(
        samples: Dict[str, Tuple[np.ndarray, np.ndarray]],
        label_names: List[str],
        out_path: str,
    ) -> None:
        """전략별 클래스 분포 막대 그래프.

        Parameters
        ──────────
        samples     {전략명: (X, y)} — y는 클래스 인덱스 배열
        label_names 클래스 이름 목록
        """
        n_cls    = len(label_names)
        colors   = plt.cm.get_cmap("tab10", n_cls).colors
        n_strat  = len(samples)
        fig, axes = plt.subplots(1, n_strat, figsize=(4 * n_strat, 5), sharey=False)
        if n_strat == 1:
            axes = [axes]
        for ax, (name, (_, y)) in zip(axes, samples.items()):
            counts = np.bincount(y, minlength=n_cls)
            bars   = ax.bar(range(n_cls), counts, color=colors)
            y_max  = max(int(counts.max()), 1)
            ax.set_ylim(0, y_max * 1.18)
            ax.set_xticks(range(n_cls))
            ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{name}\nN={len(y):,}", fontsize=8)
            ax.set_ylabel("Beat 수")
            for b, c in zip(bars, counts):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + y_max * 0.02,
                        f"{c:,}", ha="center", fontsize=5)
        fig.suptitle("전략별 클래스 분포", fontsize=12, y=1.01)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)

    @staticmethod
    def plot_embeddings(
        samples: Dict[str, Tuple[np.ndarray, np.ndarray]],
        label_names: List[str],
        out_path: str,
        max_per_strategy: int = 5_000,
        methods: Optional[List[str]] = None,
    ) -> None:
        """PCA·t-SNE·UMAP 임베딩 그리드.

        Parameters
        ──────────
        samples     {전략명: (X, y)}
        methods     ["pca", "tsne", "umap"] 중 선택. None이면 전체 (umap은 설치 시만)
        """
        if methods is None:
            methods = ["pca", "tsne"] + (["umap"] if HAS_UMAP else [])

        n_cls    = len(label_names)
        colors   = plt.cm.get_cmap("tab10", n_cls).colors
        n_strat  = len(samples)
        n_meth   = len(methods)
        fig, axes = plt.subplots(n_meth, n_strat, figsize=(4.5 * n_strat, 4 * n_meth))
        axes = np.array(axes).reshape(n_meth, n_strat)

        rng = np.random.default_rng(42)

        for col, (name, (X, y)) in enumerate(samples.items()):
            # 서브샘플
            if len(X) > max_per_strategy:
                idx = np.sort(rng.choice(len(X), max_per_strategy, replace=False))
                X, y = X[idx], y[idx]

            X50 = PCA(n_components=min(50, X.shape[1], len(X) - 1),
                      random_state=42).fit_transform(X)

            embeddings: Dict[str, np.ndarray] = {"pca": X50[:, :2]}

            if "tsne" in methods:
                perp = min(30, max(5, len(X) // 100))
                embeddings["tsne"] = TSNE(
                    2, perplexity=perp, n_iter=1000,
                    random_state=42, init="pca",
                ).fit_transform(X50)

            if "umap" in methods and HAS_UMAP:
                embeddings["umap"] = umap_lib.UMAP(2, random_state=42).fit_transform(X50)

            for row, meth in enumerate(methods):
                Z  = embeddings.get(meth)
                ax = axes[row, col]
                if Z is None:
                    ax.axis("off"); continue
                for ci in range(n_cls):
                    m = y == ci
                    if m.any():
                        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.4,
                                   color=colors[ci], label=label_names[ci])
                ax.set_title(f"{meth.upper()}\n{name}", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])

        legend = [Patch(color=colors[i], label=label_names[i]) for i in range(n_cls)]
        fig.legend(handles=legend, loc="lower center", ncol=n_cls,
                   fontsize=8, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("전략별 Beat 임베딩 분포", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)

    @staticmethod
    def plot_centroid_distances(
        samples: Dict[str, Tuple[np.ndarray, str]],
        out_path: str,
        metric: Optional[Any] = None,
    ) -> None:
        """센트로이드 거리 분포 히스토그램.

        Parameters
        ──────────
        samples   {전략명: (X_normal, plot_color)}
                  X_normal: 정상 beat waveform 배열 (N, win_len)
        metric    DistanceMetric (None이면 유클리드)
        """
        if compute_distances is None or DistanceMetric is None:
            print("[경고] math_utils 임포트 실패 — centroid distance 시각화 건너뜀")
            return

        _metric = metric or DistanceMetric.EUCLIDEAN
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, (X_norm, color) in samples.items():
            if len(X_norm) < 10:
                continue
            centroid = X_norm.mean(axis=0)
            dists    = compute_distances(X_norm, centroid, _metric)
            ax.hist(dists, bins=60, alpha=0.55, color=color,
                    label=f"{name} (N={len(X_norm):,})", density=True)
        ax.set_xlabel("센트로이드 유클리드 거리")
        ax.set_ylabel("밀도")
        ax.set_title("전략별 정상 Beat 센트로이드 거리 분포")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150); plt.close(fig)

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: List[str],
        title: str,
        out_path: str,
    ) -> None:
        """단일 클래스별 Precision/Recall/F1 히트맵 (Multi-label)."""
        n_cls = len(label_names)
        prec = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec  = recall_score(y_true, y_pred,  average=None, zero_division=0)
        f1   = f1_score(y_true, y_pred,       average=None, zero_division=0)
        data = np.stack([prec, rec, f1], axis=1)
        fig, ax = plt.subplots(figsize=(4, max(3, n_cls * 0.5 + 1.5)))
        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Prec", "Rec", "F1"], fontsize=8)
        ax.set_yticks(range(n_cls)); ax.set_yticklabels(label_names, fontsize=8)
        ax.set_title(title)
        for r in range(n_cls):
            for c in range(3):
                ax.text(c, r, f"{data[r,c]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150); plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # 학습 결과 시각화 (instance method — label_names 등 컨텍스트 필요)
    # ══════════════════════════════════════════════════════════════════════

    def plot_learning_curves(
        self,
        histories: Dict[str, Dict[str, List[float]]],
        colors:    List[str],
        out_path:  str,
    ) -> None:
        """전략별 train loss / val F1 비교 곡선.

        Parameters
        ──────────
        histories  {전략명: history_dict}  — fit() 반환값
        colors     전략 순서와 동일한 색상 리스트
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for (label, h), color in zip(histories.items(), colors):
            ep = range(1, len(h["train_loss"]) + 1)
            axes[0].plot(ep, h["train_loss"], color=color, label=label, lw=1.8)
            axes[1].plot(ep, h["val_f1"],     color=color, label=label, lw=1.8,
                         marker="o", ms=3)
        for ax, title, ylabel in zip(
            axes,
            ["Train Loss",    "Validation Macro-F1"],
            ["BCE Loss",      "Macro F1"],
        ):
            ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.suptitle("전략별 1D CNN 학습 곡선 (BCE Multi-label)", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150); plt.close(fig)

    def plot_final_metrics(
        self,
        results: Dict[str, Dict[str, Any]],
        colors:  List[str],
        out_path: str,
    ) -> None:
        """Test F1·Acc 막대그래프 + Train 크기 vs F1 데이터 효율성 산점도.

        Parameters
        ──────────
        results  {전략명: result_dict}  — keys: test_f1, test_acc, n_train
        """
        labels   = [k for k in results if results[k]]
        clrs     = [c for k, c in zip(results, colors) if results[k]]
        f1s      = [results[l]["test_f1"]  for l in labels]
        accs     = [results[l]["test_acc"] for l in labels]
        n_trains = [results[l]["n_train"]  for l in labels]
        x = np.arange(len(labels))

        fig = plt.figure(figsize=(14, 5))
        gs  = gridspec.GridSpec(1, 3)
        for i, (vals, title, ylabel) in enumerate([
            (f1s,  "Macro F1 (Test)",  "F1"),
            (accs, "Accuracy (Test)",  "Accuracy"),
        ]):
            ax = fig.add_subplot(gs[i])
            bars = ax.bar(x, vals, color=clrs, edgecolor="k", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
            ax.set_title(title); ax.set_ylabel(ylabel)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2, v + 0.004,
                        f"{v:.3f}", ha="center", fontsize=7)
            ax.set_ylim(0, min(1.0, max(vals) + 0.1))
            ax.grid(axis="y", alpha=0.3)

        ax2 = fig.add_subplot(gs[2])
        for label, color, nt, f1 in zip(labels, clrs, n_trains, f1s):
            ax2.scatter(nt, f1, color=color, s=80, edgecolors="k", zorder=3, label=label)
        ax2.set_xlabel("Train Set 크기"); ax2.set_ylabel("Macro F1")
        ax2.set_title("데이터 효율성: Train 크기 vs F1")
        ax2.legend(fontsize=6); ax2.grid(alpha=0.3)

        fig.suptitle("전략별 최종 성능 비교", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)

    def plot_confusion_matrices(
        self,
        results:  Dict[str, Dict[str, Any]],
        out_path: str,
    ) -> None:
        """전략별 클래스별 Precision / Recall / F1 히트맵 (Multi-label).

        Parameters
        ──────────
        results  {전략명: {"y_true": ..., "y_pred": ..., "test_f1": ...}}
                 y_true / y_pred 는 (N, n_classes) multi-hot int 배열
        """
        labels = [k for k in results if results[k]]
        n_strat = len(labels)
        n_cls   = self.n_classes
        col_names = ["Prec", "Rec", "F1"]

        fig, axes = plt.subplots(
            1, n_strat,
            figsize=(3.5 * n_strat, max(3, n_cls * 0.55 + 2)),
        )
        if n_strat == 1:
            axes = [axes]

        for ax, label in zip(axes, labels):
            res    = results[label]
            y_true = res["y_true"]
            y_pred = res["y_pred"]
            prec = precision_score(y_true, y_pred, average=None, zero_division=0)
            rec  = recall_score(y_true, y_pred,  average=None, zero_division=0)
            f1   = f1_score(y_true, y_pred,       average=None, zero_division=0)
            data = np.stack([prec, rec, f1], axis=1)  # (n_cls, 3)
            im   = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(col_names, fontsize=8)
            ax.set_yticks(range(n_cls))
            ax.set_yticklabels(self.label_names, fontsize=7)
            ax.set_title(f"{label}\nmacro F1={res['test_f1']:.3f}", fontsize=8)
            for r in range(n_cls):
                for c in range(3):
                    ax.text(c, r, f"{data[r,c]:.2f}", ha="center", va="center",
                            fontsize=7, color="black")
            fig.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle("클래스별 Precision / Recall / F1 (Test Set, Multi-label)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # 결과 저장
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def save_summary_csv(
        results:  Dict[str, Dict[str, Any]],
        out_path: str,
    ) -> pd.DataFrame:
        """전략별 지표 요약 CSV 저장 후 DataFrame 반환."""
        rows = [
            {
                "strategy":    k,
                "n_train":     v["n_train"],
                "best_val_f1": round(v.get("best_val_f1", 0.0), 4),
                "test_f1":     round(v["test_f1"],  4),
                "test_acc":    round(v["test_acc"],  4),
            }
            for k, v in results.items() if v
        ]
        df = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return df
