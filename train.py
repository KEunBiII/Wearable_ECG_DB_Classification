# -*- coding: utf-8 -*-
"""
train.py — ResUNet + BCEWithLogitsLoss + HiCardiDataModule

사용법:
    python train.py
    python train.py --epochs 50 --lr 1e-3 --batch 256
    python train.py --normal_ratio 2.0   # Normal : 비정상 = 2:1 언더샘플링
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor, ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from datamodule.HiCardiDataModule import HiCardiDataModule
from datamodule.utils.dataModuleUtils import DEFAULT_BEAT_CLASSES
from models import ResUNet
from trainer.LitECGMultiLabelClassifier import LitECGMultiLabelClassifier


# =============================================================================
# ResUNet 래퍼: (B,1,T) → (logits, feat)
# =============================================================================

class ResUNetEncoder(nn.Module):
    def __init__(self, num_classes: int,
                 in_channels: int = 1,
                 out_ch: int = 180,
                 mid_ch: int = 30):
        super().__init__()
        self.resunet = ResUNet(nOUT=num_classes,
                               in_channels=in_channels,
                               out_ch=out_ch, mid_ch=mid_ch)

    def forward(self, x):
        logits, x1, x2, x3, z = self.resunet(x)
        return logits, z


# =============================================================================
# 시드
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",       type=int,   default=config.MAX_EPOCHS)
    ap.add_argument("--lr",           type=float, default=config.LR)
    ap.add_argument("--batch",        type=int,   default=config.BATCH_SIZE)
    ap.add_argument("--workers",      type=int,   default=config.NUM_WORKERS)
    ap.add_argument("--seed",         type=int,   default=config.SEED)
    ap.add_argument("--out_ch",       type=int,   default=config.MODEL_OUT_CH)
    ap.add_argument("--mid_ch",       type=int,   default=config.MODEL_MID_CH)
    ap.add_argument("--threshold",    type=float, default=0.5)
    ap.add_argument("--normal_ratio", type=float, default=2.0,
                    help="Normal:비정상 비율 (언더샘플링). 0이면 비율 조정 안 함")
    ap.add_argument("--exp_name",     type=str,   default="resunet_hicardi_multilabel")
    ap.add_argument("--devices",      type=int,   default=1)
    args = ap.parse_args()

    set_seed(args.seed)

    CACHE_ROOT  = str(config.CACHE_ROOT)
    SPLIT_JSON  = str(config.SPLIT_JSON)
    RESULTS_DIR = config.RESULTS_DIR / args.exp_name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    CLASS_NAMES = list(DEFAULT_BEAT_CLASSES)   # 7클래스
    NUM_CLASSES = len(CLASS_NAMES)

    print(f"\n{'='*60}")
    print(f"  Experiment : {args.exp_name}")
    print(f"  Classes    : {NUM_CLASSES}  →  {CLASS_NAMES}")
    print(f"  Cache root : {CACHE_ROOT}")
    print(f"  Normal ratio (undersampling): {args.normal_ratio}")
    print(f"{'='*60}\n")

    # ── DataModule ────────────────────────────────────────────────────────────
    und_strategy = "random" if args.normal_ratio > 0 else "none"
    dm = HiCardiDataModule(
        split_json=SPLIT_JSON,
        return_arg={
            "waveform":    True,
            "labels":      True,
            "mode":        "beat",
            "n_classes":   NUM_CLASSES,
            "label_names": CLASS_NAMES,
        },
        undersampling_arg={
            "strategy":     und_strategy,
            "normal_ratio": args.normal_ratio,
            "apply_to_train": True,
            "apply_to_val":   False,
            "apply_to_test":  False,
        },
        dataloader_arg={
            "batch_size":  args.batch,
            "num_workers": args.workers,
            "pin_memory":  True,
            "drop_last":   True,
        },
    )
    dm.setup()
    dm.print_balance_summary()

    # bce_dataloader: (waveform, multi-hot_labels) 반환
    train_dl = dm.bce_dataloader("train", args.batch, args.workers)
    val_dl   = dm.bce_dataloader("val",   args.batch, args.workers)

    # ── 모델 ─────────────────────────────────────────────────────────────────
    encoder = ResUNetEncoder(
        num_classes=NUM_CLASSES,
        in_channels=config.MODEL_IN_CHANNELS,
        out_ch=args.out_ch,
        mid_ch=args.mid_ch,
    )

    lit_model = LitECGMultiLabelClassifier(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
        lr=args.lr,
        threshold=args.threshold,
        export_dir=str(RESULTS_DIR),
        verbose=True,
    )

    # ── 콜백 ─────────────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath=str(RESULTS_DIR / "checkpoints"),
            filename="epoch{epoch:03d}-f1{val/macro_f1:.4f}",
            monitor="val/macro_f1",
            mode="max",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── 로거 ─────────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(
        save_dir=str(config.RESULTS_DIR),
        name=args.exp_name,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
    )

    trainer.fit(lit_model, train_dl, val_dl)
    print(f"\n[Done] best checkpoint: {trainer.checkpoint_callback.best_model_path}")

    # ── Test ─────────────────────────────────────────────────────────────────
    test_dl = dm.bce_dataloader("test", args.batch, args.workers)
    trainer.test(lit_model, test_dl, ckpt_path="best")


if __name__ == "__main__":
    main()
