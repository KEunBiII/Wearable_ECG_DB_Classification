# -*- coding: utf-8 -*-
"""
test.py — 저장된 체크포인트로 test set 평가

사용법:
    python test.py --ckpt results/resunet_hicardi_multilabel/checkpoints/epoch002-f10.6296.ckpt
    python test.py --ckpt results/resunet_hicardi_multilabel/checkpoints/epoch002-f10.6296.ckpt --batch 512
    python test.py --ckpt ... --exclude_classes Trigeminy
    python test.py --ckpt ... --exclude_classes Trigeminy APC
"""

import argparse
import sys
from pathlib import Path

import torch
import lightning as L

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from datamodule.HiCardiDataModule import HiCardiDataModule
from datamodule.utils.dataModuleUtils import DEFAULT_BEAT_CLASSES
from models import ResUNet
from train import ResUNetEncoder
from trainer.LitECGMultiLabelClassifier import LitECGMultiLabelClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",    type=str, required=True, help="체크포인트 경로")
    ap.add_argument("--batch",   type=int, default=config.BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--out_ch",  type=int, default=config.MODEL_OUT_CH)
    ap.add_argument("--mid_ch",  type=int, default=config.MODEL_MID_CH)
    ap.add_argument("--threshold",      type=float, default=0.5)
    ap.add_argument("--exp_name",       type=str,   default="resunet_hicardi_multilabel")
    ap.add_argument("--exclude_classes", nargs="+", default=[], help="평가에서 제외할 클래스명 (예: Trigeminy APC)")
    args = ap.parse_args()

    CLASS_NAMES = list(DEFAULT_BEAT_CLASSES)
    NUM_CLASSES = len(CLASS_NAMES)

    # 평가용 클래스 (모델 구조는 그대로, 지표 계산 시에만 제외)
    EVAL_CLASSES = [c for c in CLASS_NAMES if c not in args.exclude_classes]
    if args.exclude_classes:
        print(f"  제외 클래스 : {args.exclude_classes}")
        print(f"  평가 클래스 : {EVAL_CLASSES}")
    RESULTS_DIR = config.RESULTS_DIR / args.exp_name

    print(f"\n{'='*60}")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  Classes    : {NUM_CLASSES}  →  {CLASS_NAMES}")
    print(f"{'='*60}\n")

    # ── DataModule ────────────────────────────────────────────────────────────
    dm = HiCardiDataModule(
        split_json=str(config.SPLIT_JSON),
        return_arg={
            "waveform":    True,
            "labels":      True,
            "mode":        "beat",
            "n_classes":   NUM_CLASSES,
            "label_names": CLASS_NAMES,
        },
        dataloader_arg={
            "batch_size":  args.batch,
            "num_workers": args.workers,
            "pin_memory":  True,
        },
    )
    dm.setup()
    test_dl = dm.bce_dataloader("test", args.batch, args.workers)

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    encoder = ResUNetEncoder(
        num_classes=NUM_CLASSES,
        in_channels=config.MODEL_IN_CHANNELS,
        out_ch=args.out_ch,
        mid_ch=args.mid_ch,
    )
    lit_model = LitECGMultiLabelClassifier.load_from_checkpoint(
        args.ckpt,
        encoder=encoder,
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
        threshold=args.threshold,
        export_dir=str(RESULTS_DIR),
        eval_class_names=EVAL_CLASSES,
    )

    # ── Test ─────────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
    )
    trainer.test(lit_model, test_dl)


if __name__ == "__main__":
    main()
