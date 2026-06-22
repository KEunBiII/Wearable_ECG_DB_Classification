import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import xlsxwriter

from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,
    MulticlassAUROC, MulticlassAveragePrecision, MulticlassF1Score
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, f1_score, precision_score,
    recall_score, accuracy_score
)
import numpy as np
import pandas as pd
import warnings

from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")


class LitECGClassifier(L.LightningModule):
    def __init__(self, encoder, num_classes=5, feat_dim=256, lr=1e-3,
                 export_dir: str = "results",
                 class_names_full=("N", "S", "V", "F", "Q"),
                 verbose: bool = False,
                 optimizer_name: str = "AdamW",
                 scheduler_name: str = "CosineAnnealingLR"):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.encoder = encoder
        self.criterion_cls = nn.CrossEntropyLoss()

        self.val_acc   = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.val_prec  = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.val_rec   = MulticlassRecall(num_classes=num_classes, average='macro')
        self.val_f1    = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.val_auroc = MulticlassAUROC(num_classes=num_classes, average='macro')
        self.val_auprc = MulticlassAveragePrecision(num_classes=num_classes, average='macro')

        self.best_val_f1      = 0.0
        self.best_epoch       = 0
        self.best_val_metrics = {}
        self.best_snapshot    = None
        self.verbose          = verbose

        self.val_preds, self.val_probs, self.val_targets = [], [], []

        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.class_names_full = list(class_names_full)

    # ----------------------------------
    def forward(self, x):
        # Accept inputs of shape [B, C, T]. For beat-classifier C==1.
        # Some datamodules may provide a context sequence shape [B, K, T]
        # (K>1). In that case select the center beat as classifier input.
        if x.ndim == 3 and x.shape[1] != 1:
            # assume x is (B, K, T) -> pick center beat
            center = x.shape[1] // 2
            x = x[:, center:center+1, :]

        # Try standard forward if available, else fallback to specific methods
        try:
            # For models like ResUMamba2 that return (logits, feat)
            y_logits, feat = self.encoder(x)
        except (TypeError, ValueError, AttributeError):
            try:
                # Legacy / Specific architecture fallback
                feat = self.encoder.forward_resu_features(x)
                y_logits = self.encoder.fc(feat)
            except AttributeError:
                # Basic encoder-only forward
                feat = self.encoder(x)
                if hasattr(self.encoder, "fc"):
                    y_logits = self.encoder.fc(feat)
                else:
                    y_logits = feat # assuming encoder is the full model

        return y_logits, feat

    # ----------------------------------
    def training_step(self, batch, batch_idx):
        # 1) Key selection (waveform vs waveforms)
        x = batch.get("waveform")
        if x is None:
            x = batch.get("waveforms")
        if x is None:
            raise KeyError("Batch must contain 'waveform' or 'waveforms'")

        # 2) Shape adjustment: (B, T) -> (B, 1, T)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        # if x.ndim == 3 and shape[1] > 1, forward() will pick center
        y = batch["labels"]                       # (B, n_cls) multi-hot

        # multi-hot → class index for CrossEntropyLoss
        try:
            y_proc = y.argmax(dim=1).long()
        except Exception:
            y_proc = y.long()

        # Filter out invalid labels (negative), if any
        mask = (y_proc >= 0)
        if mask.numel() == 0 or mask.sum().item() == 0:
            # Nothing to train on in this batch
            self.log("train/skip_batch", 1, prog_bar=False)
            return torch.tensor(0.0, device=self.device)

        if mask.sum().item() != mask.numel():
            x = x[mask]
            y_proc = y_proc[mask]

        # move target to the same device as inputs/params and forward
        y_proc = y_proc.to(self.device)
        x = x.to(self.device)

        y_logits, _ = self(x)

        loss = self.criterion_cls(y_logits, y_proc)
        acc  = (y_logits.argmax(1) == y_proc).float().mean()
        self.log_dict({"train/loss": loss, "train/acc": acc}, prog_bar=True)
        return loss

    # ----------------------------------
    def validation_step(self, batch, batch_idx):
        x = batch.get("waveform")
        if x is None:
            x = batch.get("waveforms")
        if x is None:
            return # or raise error

        if x.ndim == 2:
            x = x.unsqueeze(1)
        y = batch["labels"]

        # multi-hot → class index for CrossEntropyLoss
        try:
            y_proc = y.argmax(dim=1).long()
        except Exception:
            y_proc = y.long()

        # Filter out invalid labels (negative)
        mask = (y_proc >= 0)
        if mask.numel() == 0 or mask.sum().item() == 0:
            # Nothing to validate in this batch
            self.log("val/skip_batch", 1, prog_bar=False)
            return torch.tensor(0.0, device=self.device)

        if mask.sum().item() != mask.numel():
            x = x[mask]
            y_proc = y_proc[mask]

        # move tensors to device
        y_proc = y_proc.to(self.device)
        x = x.to(self.device)

        # Validate targets on CPU to avoid CUDA device-side asserts
        try:
            y_cpu = y_proc.cpu()
        except Exception:
            y_cpu = y_proc.detach().to('cpu')

        n_classes = int(self.hparams.num_classes) if hasattr(self.hparams, 'num_classes') else None
        if n_classes is not None:
            if y_cpu.numel() > 0:
                min_lbl = int(y_cpu.min())
                max_lbl = int(y_cpu.max())
                if min_lbl < 0 or max_lbl >= n_classes:
                    uniq = np.unique(y_cpu.numpy())[:20]
                    msg = (
                        f"Invalid target labels in validation batch {batch_idx}: "
                        f"min={min_lbl}, max={max_lbl}, n_classes={n_classes}, unique(sample)={uniq}"
                    )
                    print(msg)
                    print(f"y_logits.shape will be computed after forward; y_proc.shape={tuple(y_proc.shape)}")
                    raise RuntimeError(msg)

        y_logits, _ = self(x)

        loss = self.criterion_cls(y_logits, y_proc)

        probs = F.softmax(y_logits, dim=1)
        preds = probs.argmax(1)

        self.val_preds.append(preds.cpu())
        self.val_probs.append(probs.cpu())
        self.val_targets.append(y_proc.cpu())

        self.log("val/loss_cls", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if not self.val_targets:
            if self.verbose:
                print("[Validation] No batches to evaluate.")
            return

        y_true   = torch.cat(self.val_targets).numpy()
        y_pred   = torch.cat(self.val_preds).numpy()
        probs_t  = torch.cat(self.val_probs).numpy()
        self.val_targets.clear(); self.val_preds.clear(); self.val_probs.clear()

        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        # Ensure unique_classes are valid indices for probs_t
        unique_classes = unique_classes[unique_classes < probs_t.shape[1]]
        class_names    = [self.class_names_full[i] for i in unique_classes]

        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        TN = cm.sum() - (cm.sum(1) + cm.sum(0) - np.diag(cm))
        FP = cm.sum(0) - np.diag(cm)
        FN = cm.sum(1) - np.diag(cm)
        TP = np.diag(cm)
        specificity = TN / (TN + FP + 1e-12)
        sensitivity = TP / (TP + FN + 1e-12)

        report    = classification_report(
            y_true, y_pred, labels=unique_classes, target_names=class_names,
            output_dict=True, zero_division=0
        )
        df_report = pd.DataFrame(report).T

        selected_probs = probs_t[:, unique_classes]

        try:
            macro_auc = roc_auc_score(y_true, selected_probs,
                                      multi_class="ovr", average="macro",
                                      labels=unique_classes)
            macro_prc = average_precision_score(
                label_binarize(y_true, classes=unique_classes),
                selected_probs, average="macro"
            )
        except ValueError as e:
            if self.verbose:
                print(f"[Warning] Skipping ROC/AUPRC: {e}")
            macro_auc, macro_prc = np.nan, np.nan

        macro_f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_spec = np.nanmean(specificity)
        macro_acc  = accuracy_score(y_true, y_pred)

        if self.verbose:
            print(
                f"\n[Validation | FP32 fine-tune | NOT PTQ] Epoch {self.current_epoch}"
            )
            print(df_report.round(4))
            print(
                f"Macro AUROC:{macro_auc:.4f} AUPRC:{macro_prc:.4f} F1:{macro_f1:.4f} "
                f"Spec:{macro_spec:.4f}  ← Trainer.fit 내부, 양자화 전\n"
            )

        if self.logger is not None and hasattr(self.logger, "experiment"):
            tb = self.logger.experiment
            tb.add_scalar("Metrics/Macro_AUROC",        macro_auc,  self.current_epoch)
            tb.add_scalar("Metrics/Macro_AUPRC",        macro_prc,  self.current_epoch)
            tb.add_scalar("Metrics/Macro_F1",           macro_f1,   self.current_epoch)
            tb.add_scalar("Metrics/Macro_Sensitivity",  macro_rec,  self.current_epoch)
            tb.add_scalar("Metrics/Macro_Specificity",  macro_spec, self.current_epoch)
            tb.add_scalar("Metrics/Macro_Accuracy",     macro_acc,  self.current_epoch)
            for i, cls_name in enumerate(class_names):
                tb.add_scalar(f"PerClass/{cls_name}_Sensitivity", sensitivity[i], self.current_epoch)
                tb.add_scalar(f"PerClass/{cls_name}_Specificity", specificity[i], self.current_epoch)
                tb.add_scalar(f"PerClass/{cls_name}_F1", df_report.loc[cls_name, "f1-score"], self.current_epoch)

        if macro_f1 > self.best_val_f1:
            self.best_val_f1  = macro_f1
            self.best_epoch   = self.current_epoch
            self.best_val_metrics = {
                "Macro_F1":          macro_f1,
                "Macro_AUROC":       macro_auc,
                "Macro_AUPRC":       macro_prc,
                "Macro_Precision":   macro_prec,
                "Macro_Recall":      macro_rec,
                "Macro_Specificity": macro_spec,
                "Accuracy":          macro_acc,
            }
            df_per_class, cm_best = self._build_per_class_table(
                y_true, y_pred, selected_probs, unique_classes, class_names
            )
            df_avg = self._macro_micro_weighted_block(
                y_true, y_pred, selected_probs, unique_classes, cm_best
            )
            self.best_snapshot = {
                "y_true": y_true, "y_pred": y_pred,
                "probs": selected_probs,
                "unique_classes": unique_classes, "class_names": class_names,
                "df_per_class": df_per_class, "df_avg": df_avg, "cm": cm_best
            }

        self.log("val/macro_f1",    macro_f1,   prog_bar=True)
        self.log("val/macro_spec",  macro_spec, prog_bar=True)
        self.log("val/macro_rec",   macro_rec,  prog_bar=True)
        self.log("val/macro_auroc", macro_auc,  prog_bar=False, on_epoch=True)
        self.log("val/macro_auprc", macro_prc,  prog_bar=False, on_epoch=True)

    # ----------------------------------
    def _safe_auc_pr_per_class(self, y_true, probs, unique_classes):
        n = len(unique_classes)
        auroc_per_cls = np.full(n, np.nan, dtype=float)
        auprc_per_cls = np.full(n, np.nan, dtype=float)
        for i, c in enumerate(unique_classes):
            y_bin = (y_true == c).astype(int)
            p = probs[:, i]
            if y_bin.max() == 1 and y_bin.min() == 0:
                try:
                    auroc_per_cls[i] = roc_auc_score(y_bin, p)
                except ValueError:
                    pass
                try:
                    auprc_per_cls[i] = average_precision_score(y_bin, p)
                except ValueError:
                    pass
        return auroc_per_cls, auprc_per_cls

    def _macro_micro_weighted_block(self, y_true, y_pred, probs, unique_classes, cm):
        support_vec = np.array([(y_true == c).sum() for c in unique_classes], dtype=int)
        total = support_vec.sum()

        TN = cm.sum() - (cm.sum(1) + cm.sum(0) - np.diag(cm))
        FP = cm.sum(0) - np.diag(cm)
        FN = cm.sum(1) - np.diag(cm)
        TP = np.diag(cm)
        spec_per_cls = TN / (TN + FP + 1e-12)

        prec_macro    = precision_score(y_true, y_pred, average="macro",    zero_division=0)
        rec_macro     = recall_score(y_true,  y_pred,   average="macro",    zero_division=0)
        f1_macro      = f1_score(y_true,      y_pred,   average="macro",    zero_division=0)
        prec_micro    = precision_score(y_true, y_pred, average="micro",    zero_division=0)
        rec_micro     = recall_score(y_true,  y_pred,   average="micro",    zero_division=0)
        f1_micro      = f1_score(y_true,      y_pred,   average="micro",    zero_division=0)
        prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec_weighted  = recall_score(y_true,  y_pred,   average="weighted", zero_division=0)
        f1_weighted   = f1_score(y_true,      y_pred,   average="weighted", zero_division=0)

        acc_overall = accuracy_score(y_true, y_pred)
        acc_macro   = np.mean((TP + TN) / (TP + TN + FP + FN + 1e-12))

        spec_macro    = np.nanmean(spec_per_cls)
        spec_weighted = np.nansum(spec_per_cls * (support_vec / (total + 1e-12)))
        TP_tot = TP.sum(); FP_tot = FP.sum(); FN_tot = FN.sum()
        TN_tot = total - TP_tot - FP_tot - FN_tot
        spec_micro = TN_tot / (TN_tot + FP_tot + 1e-12)

        y_true_onehot = label_binarize(y_true, classes=unique_classes)
        P = probs

        def _safe_roc_auc(avg):
            try:
                return roc_auc_score(y_true, P, multi_class="ovr", average=avg)
            except Exception:
                return np.nan

        def _safe_ap(avg):
            try:
                return average_precision_score(y_true_onehot, P, average=avg)
            except Exception:
                return np.nan

        rows = [
            ["macro",    _safe_roc_auc("macro"),    _safe_ap("macro"),    rec_macro,    spec_macro,    prec_macro,    f1_macro,    acc_macro],
            ["micro",    _safe_roc_auc("micro"),    _safe_ap("micro"),    rec_micro,    spec_micro,    prec_micro,    f1_micro,    acc_overall],
            ["weighted", _safe_roc_auc("weighted"), _safe_ap("weighted"), rec_weighted, spec_weighted, prec_weighted, f1_weighted, acc_overall],
        ]
        return pd.DataFrame(rows, columns=["average","auroc","auprc","sensitivity","specificity","precision","f1","accuracy"])

    def _build_per_class_table(self, y_true, y_pred, probs, unique_classes, class_names):
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        TN = cm.sum() - (cm.sum(1) + cm.sum(0) - np.diag(cm))
        FP = cm.sum(0) - np.diag(cm)
        FN = cm.sum(1) - np.diag(cm)
        TP = np.diag(cm)

        support = (y_true.reshape(-1, 1) == unique_classes.reshape(1, -1)).sum(axis=0)
        sens = TP / (TP + FN + 1e-12)
        spec = TN / (TN + FP + 1e-12)
        prec = np.divide(TP, (TP + FP + 1e-12))
        f1   = np.divide(2 * prec * sens, (prec + sens + 1e-12))
        acc_cls = (TP + TN) / (TP + TN + FP + FN + 1e-12)

        auroc_per_cls, auprc_per_cls = self._safe_auc_pr_per_class(y_true, probs, unique_classes)

        df = pd.DataFrame({
            "class_id":    unique_classes,
            "class":       [class_names[i] for i in range(len(unique_classes))],
            "support":     support.astype(int),
            "sensitivity": sens,
            "specificity": spec,
            "precision":   prec,
            "f1":          f1,
            "accuracy":    acc_cls,
            "auroc":       auroc_per_cls,
            "auprc":       auprc_per_cls,
        })
        return df, cm

    def _export_to_excel(self, df_per_class, df_avg, cm, unique_classes, class_names, meta):
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name  = self.logger.name    if self.logger is not None else "NoLogger"
        run_ver   = str(self.logger.version) if self.logger is not None else "v0"
        fname     = self.export_dir / f"{run_name}_{run_ver}_epoch{meta['best_epoch']}.xlsx"

        with pd.ExcelWriter(fname, engine="xlsxwriter") as writer:
            for col in ["auroc","auprc","sensitivity","specificity","precision","f1","accuracy"]:
                df_avg[col] = df_avg[col].astype(float).round(6)
            df_avg.to_excel(writer, sheet_name="Averages", index=False)

            for col in ["sensitivity","specificity","precision","f1","accuracy","auroc","auprc"]:
                df_per_class[col] = df_per_class[col].astype(float).round(6)
            df_per_class.to_excel(writer, sheet_name="PerClass", index=False)

            df_cm = pd.DataFrame(cm,
                index=[class_names[i] for i in range(len(unique_classes))],
                columns=[class_names[i] for i in range(len(unique_classes))])
            df_cm.to_excel(writer, sheet_name="ConfusionMatrix")

            pd.DataFrame(list(meta.items()), columns=["key","value"]).to_excel(
                writer, sheet_name="Metadata", index=False)
        print(f"[Export] Wrote Excel to: {fname}")

    def on_train_end(self):
        print(f"\nBest epoch: {self.best_epoch}, F1={self.best_val_f1:.4f}")
        for k, v in self.best_val_metrics.items():
            print(f"  {k}: {v:.4f}")
            if self.logger is not None and hasattr(self.logger, "experiment"):
                try:
                    self.logger.experiment.add_scalar(f"Final/Best_{k}", v, self.best_epoch)
                except Exception:
                    pass

        if self.best_snapshot is not None:
            meta = {"best_epoch": int(self.best_epoch)}
            meta.update({k: float(v) if isinstance(v, (int, float, np.floating)) else v
                         for k, v in self.best_val_metrics.items()})
            self._export_to_excel(
                self.best_snapshot["df_per_class"], self.best_snapshot["df_avg"],
                self.best_snapshot["cm"], self.best_snapshot["unique_classes"],
                self.best_snapshot["class_names"], meta
            )
        else:
            print("[Export] No best snapshot captured (did validation run?).")

    # ----------------------------------
    def configure_optimizers(self):
        opt_name   = getattr(self.hparams, "optimizer_name",  "AdamW")
        sched_name = getattr(self.hparams, "scheduler_name",  "CosineAnnealingLR")
        lr         = self.hparams.lr
        wd         = 1e-2

        if opt_name == "Adam":
            opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "RMSprop":
            opt = torch.optim.RMSprop(self.parameters(), lr=lr, weight_decay=wd)
        else:
            opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

        if sched_name == "StepLR":
            step = max(1, self.trainer.max_epochs // 3)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step, gamma=0.1)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        elif sched_name == "OneCycleLR":
            try:
                total_steps = int(self.trainer.estimated_stepping_batches)
            except Exception:
                total_steps = self.trainer.max_epochs * 500
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=lr * 10, total_steps=total_steps
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
