# -*- coding: utf-8 -*-
"""
LitECGMultiLabelClassifier.py
— BCEWithLogitsLoss + multi-label 평가지표 (per-class F1/AUROC/AUPRC)
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


class LitECGMultiLabelClassifier(L.LightningModule):
    """
    ResUNet 래퍼 + BCEWithLogitsLoss + multi-label 평가.

    encoder : (B,1,T) → (logits (B,C), feat (B,D))
    labels  : (B,C) float multi-hot  (bce_dataloader 출력)
    """

    def __init__(
        self,
        encoder:          nn.Module,
        num_classes:      int,
        class_names:      List[str],
        lr:               float = 1e-3,
        threshold:        float = 0.5,
        export_dir:       str   = "results",
        verbose:          bool  = True,
        eval_class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.encoder          = encoder
        self.criterion        = nn.BCEWithLogitsLoss()
        self.threshold        = threshold
        self.class_names      = class_names
        self.eval_class_names = eval_class_names or class_names  # 평가 시 사용할 클래스
        self.export_dir       = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.verbose          = verbose

        self._train_logits: list = []
        self._train_labels: list = []
        self._val_logits: list  = []
        self._val_labels: list  = []
        self._test_logits: list = []
        self._test_labels: list = []
        self.best_val_macro_f1  = 0.0
        self.best_epoch         = 0
        self.best_metrics: dict = {}

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        logits, feat = self.encoder(x)
        return logits, feat

    # ------------------------------------------------------------------
    def training_step(self, batch, _):
        x, y = batch                           # x:(B,1,T)  y:(B,C) float
        if x.ndim == 2:
            x = x.unsqueeze(1)
        logits, _ = self(x)
        loss = self.criterion(logits, y)
        preds = (torch.sigmoid(logits) > self.threshold).float()
        acc = (preds == y).all(dim=1).float().mean()
        self._train_logits.append(logits.detach().cpu())
        self._train_labels.append(y.detach().cpu())
        self.log_dict({"train/loss": loss, "train/acc": acc}, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if not self._train_logits:
            return
        logits = torch.cat(self._train_logits).numpy()
        labels = torch.cat(self._train_labels).numpy()
        self._train_logits.clear()
        self._train_labels.clear()

        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > self.threshold).astype(int)
        metrics = self._compute_metrics(labels, preds, probs)

        self.log("train/macro_auroc",    metrics["macro"]["auroc"],    prog_bar=False)
        self.log("train/macro_auprc",    metrics["macro"]["auprc"],    prog_bar=False)
        self.log("train/weighted_auroc", metrics["weighted"]["auroc"], prog_bar=False)
        self.log("train/weighted_auprc", metrics["weighted"]["auprc"], prog_bar=False)

    # ------------------------------------------------------------------
    def validation_step(self, batch, _):
        x, y = batch
        if x.ndim == 2:
            x = x.unsqueeze(1)
        logits, _ = self(x)
        loss = self.criterion(logits, y)
        self._val_logits.append(logits.detach().cpu())
        self._val_labels.append(y.detach().cpu())
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return

        logits = torch.cat(self._val_logits).numpy()   # (N, C)
        labels = torch.cat(self._val_labels).numpy()   # (N, C)
        self._val_logits.clear()
        self._val_labels.clear()

        probs = 1 / (1 + np.exp(-logits))             # sigmoid
        preds = (probs > self.threshold).astype(int)

        metrics = self._compute_metrics(labels, preds, probs)

        # TensorBoard 로깅
        if self.logger is not None and hasattr(self.logger, "experiment"):
            tb = self.logger.experiment
            for k, v in metrics["macro"].items():
                if np.isfinite(v):
                    tb.add_scalar(f"Metrics/macro_{k}", v, self.current_epoch)
            for cls, row in metrics["per_class"].items():
                for k, v in row.items():
                    if np.isfinite(v):
                        tb.add_scalar(f"PerClass/{cls}/{k}", v, self.current_epoch)

        macro_f1 = metrics["macro"]["f1"]
        self.log("val/macro_f1",       macro_f1,                        prog_bar=True)
        self.log("val/macro_auroc",    metrics["macro"]["auroc"],        prog_bar=True)
        self.log("val/macro_auprc",    metrics["macro"]["auprc"],        prog_bar=False)
        self.log("val/macro_sens",     metrics["macro"]["sens"],         prog_bar=False)
        self.log("val/macro_spec",     metrics["macro"]["spec"],         prog_bar=False)
        self.log("val/weighted_auroc", metrics["weighted"]["auroc"],     prog_bar=False)
        self.log("val/weighted_auprc", metrics["weighted"]["auprc"],     prog_bar=False)

        if self.verbose:
            self._print_metrics(metrics)

        if macro_f1 > self.best_val_macro_f1:
            self.best_val_macro_f1 = macro_f1
            self.best_epoch        = self.current_epoch
            self.best_metrics      = metrics

    # ------------------------------------------------------------------
    def test_step(self, batch, _):
        x, y = batch
        if x.ndim == 2:
            x = x.unsqueeze(1)
        logits, _ = self(x)
        loss = self.criterion(logits, y)
        self._test_logits.append(logits.detach().cpu())
        self._test_labels.append(y.detach().cpu())
        self.log("test/loss", loss, prog_bar=True, on_epoch=True)

    def on_test_epoch_end(self):
        if not self._test_logits:
            return

        logits = torch.cat(self._test_logits).numpy()
        labels = torch.cat(self._test_labels).numpy()
        self._test_logits.clear()
        self._test_labels.clear()

        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > self.threshold).astype(int)

        # eval_class_names에 해당하는 컬럼만 추려서 평가
        eval_idx = [self.class_names.index(n) for n in self.eval_class_names
                    if n in self.class_names]
        metrics = self._compute_metrics(
            labels[:, eval_idx], preds[:, eval_idx], probs[:, eval_idx],
            class_names=self.eval_class_names,
        )

        self.log("test/macro_f1",       metrics["macro"]["f1"])
        self.log("test/macro_auroc",    metrics["macro"]["auroc"])
        self.log("test/macro_auprc",    metrics["macro"]["auprc"])
        self.log("test/macro_sens",     metrics["macro"]["sens"])
        self.log("test/macro_spec",     metrics["macro"]["spec"])
        self.log("test/weighted_auroc", metrics["weighted"]["auroc"])
        self.log("test/weighted_auprc", metrics["weighted"]["auprc"])

        self._print_metrics(metrics, split="Test")
        self._export_excel(metrics, prefix="test")

    # ------------------------------------------------------------------
    def _compute_metrics(self, labels, preds, probs, class_names: Optional[List[str]] = None):
        names = class_names if class_names is not None else self.class_names
        per_class = {}

        for i, name in enumerate(names):
            y_t = labels[:, i]
            y_p = preds[:, i]
            y_s = probs[:, i]

            if y_t.sum() == 0:              # 해당 클래스 샘플 없음
                per_class[name] = {k: float("nan") for k in
                                   ["f1","sens","spec","auroc","auprc","support"]}
                continue

            cm = confusion_matrix(y_t, y_p, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            sens  = tp / (tp + fn + 1e-12)
            spec  = tn / (tn + fp + 1e-12)
            f1    = f1_score(y_t, y_p, zero_division=0)
            try:
                auroc = roc_auc_score(y_t, y_s)
            except Exception:
                auroc = float("nan")
            try:
                auprc = average_precision_score(y_t, y_s)
            except Exception:
                auprc = float("nan")

            per_class[name] = {
                "f1": f1, "sens": sens, "spec": spec,
                "auroc": auroc, "auprc": auprc,
                "support": int(y_t.sum()),
            }

        # macro 평균 (nan 제외)
        def _nanmean(key):
            vals = [v[key] for v in per_class.values() if np.isfinite(v[key])]
            return float(np.mean(vals)) if vals else float("nan")

        # weighted 평균 (support 기준 가중)
        def _nanweighted(key):
            pairs = [(v[key], v["support"]) for v in per_class.values()
                     if np.isfinite(v[key]) and np.isfinite(v["support"])]
            if not pairs:
                return float("nan")
            total = sum(s for _, s in pairs)
            return float(sum(v * s for v, s in pairs) / total) if total > 0 else float("nan")

        macro    = {k: _nanmean(k)     for k in ["f1", "sens", "spec", "auroc", "auprc"]}
        weighted = {k: _nanweighted(k) for k in ["f1", "sens", "spec", "auroc", "auprc"]}
        return {"per_class": per_class, "macro": macro, "weighted": weighted}

    def _print_metrics(self, metrics, split: str = "Val"):
        print(f"\n[{split} | Epoch {self.current_epoch}]")
        header = f"  {'Class':<14} {'F1':>6} {'Sens':>6} {'Spec':>6} {'AUROC':>6} {'AUPRC':>6} {'N':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, row in metrics["per_class"].items():
            if row["support"] != row["support"]:  # nan
                continue
            print(f"  {name:<14} "
                  f"{row['f1']:>6.3f} {row['sens']:>6.3f} {row['spec']:>6.3f} "
                  f"{row['auroc']:>6.3f} {row['auprc']:>6.3f} {row['support']:>7,}")
        m = metrics["macro"]
        print(f"  {'macro':<14} "
              f"{m['f1']:>6.3f} {m['sens']:>6.3f} {m['spec']:>6.3f} "
              f"{m['auroc']:>6.3f} {m['auprc']:>6.3f}")

    # ------------------------------------------------------------------
    def on_train_end(self):
        print(f"\nBest epoch: {self.best_epoch}  macro_F1={self.best_val_macro_f1:.4f}")
        if self.best_metrics:
            self._export_excel(self.best_metrics)

    def _export_excel(self, metrics, prefix: str = "best"):
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix == "best":
            fname = self.export_dir / f"best_epoch{self.best_epoch}_{ts}.xlsx"
        else:
            fname = self.export_dir / f"{prefix}_{ts}.xlsx"
        rows  = []
        for name, row in metrics["per_class"].items():
            rows.append({"class": name, **row})
        df_pc = pd.DataFrame(rows)
        df_mc = pd.DataFrame([{"type": "macro", **metrics["macro"]}])
        df_wc = pd.DataFrame([{"type": "weighted", **metrics["weighted"]}])
        with pd.ExcelWriter(fname, engine="xlsxwriter") as w:
            df_pc.to_excel(w, sheet_name="PerClass",  index=False)
            df_mc.to_excel(w, sheet_name="Macro",     index=False)
            df_wc.to_excel(w, sheet_name="Weighted",  index=False)
        print(f"[Export] {fname}")

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
