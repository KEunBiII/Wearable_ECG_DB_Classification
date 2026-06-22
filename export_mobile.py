# -*- coding: utf-8 -*-
"""
export_mobile.py — 학습된 .ckpt → ExecuTorch .pte 변환

사용법:
    python export_mobile.py --ckpt results/demo_model/checkpoints/best.ckpt
    python export_mobile.py --ckpt results/demo_model/checkpoints/best.ckpt --out results/model.pte
    python export_mobile.py --ckpt results/demo_model/checkpoints/best.ckpt --verify

입력: (1, 1, 501) float32  — 1 beat, z-score 정규화 완료된 신호
출력: (1, 7) float32        — 7개 클래스 logit (Sigmoid 전)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from models import ResUNet
from datamodule.utils.dataModuleUtils import DEFAULT_BEAT_CLASSES


# =============================================================================
# 추론 전용 래퍼 (logits만 반환)
# =============================================================================

class ECGClassifier(nn.Module):
    """Export 전용: (1, 1, 501) → logits (1, n_classes)"""

    def __init__(self, num_classes: int = 7,
                 out_ch: int = config.MODEL_OUT_CH,
                 mid_ch: int = config.MODEL_MID_CH):
        super().__init__()
        self.resunet = ResUNet(
            nOUT=num_classes,
            in_channels=config.MODEL_IN_CHANNELS,
            out_ch=out_ch,
            mid_ch=mid_ch,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, *_ = self.resunet(x)
        return logits


# =============================================================================
# 체크포인트 로드
# =============================================================================

def load_model(ckpt_path: str, num_classes: int = 7,
               out_ch: int = config.MODEL_OUT_CH,
               mid_ch: int = config.MODEL_MID_CH) -> ECGClassifier:
    """
    LightningModule .ckpt에서 ResUNet 가중치만 추출해 ECGClassifier로 반환.
    state_dict 키 구조: encoder.resunet.* → resunet.*
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = ckpt["state_dict"]

    model = ECGClassifier(num_classes=num_classes, out_ch=out_ch, mid_ch=mid_ch)

    remapped = {
        k.replace("encoder.resunet.", "resunet."): v
        for k, v in raw_sd.items()
        if k.startswith("encoder.resunet.")
    }
    missing, unexpected = model.load_state_dict(remapped, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch — missing: {missing}, unexpected: {unexpected}")

    model.eval()
    return model


# =============================================================================
# 내보내기
# =============================================================================

def export_pte(ckpt_path: str, out_path: str,
               num_classes: int = 7,
               out_ch: int = config.MODEL_OUT_CH,
               mid_ch: int = config.MODEL_MID_CH,
               use_xnnpack: bool = True,
               verify: bool = False):

    print(f"\n{'='*60}")
    print(f"  ckpt       : {ckpt_path}")
    print(f"  out        : {out_path}")
    print(f"  num_classes: {num_classes}  {list(DEFAULT_BEAT_CLASSES)}")
    print(f"  XNNPACK    : {use_xnnpack}")
    print(f"{'='*60}\n")

    # ── 1. 모델 로드 ─────────────────────────────────────────────────────────
    print("[1/5] 체크포인트 로드 중...")
    model = load_model(ckpt_path, num_classes, out_ch, mid_ch)
    example_input = torch.randn(1, 1, 501)

    # ── 2. 사전 검증 ─────────────────────────────────────────────────────────
    with torch.no_grad():
        ref_output = model(example_input)
    print(f"      원본 모델 출력 shape: {ref_output.shape}")

    # ── 3. torch.export ───────────────────────────────────────────────────────
    print("[2/5] torch.export.export() ...")
    # eval() + no_grad: BatchNorm이 inference 경로(running stats)로 트레이스되도록
    model.eval()
    with torch.no_grad():
        exported = torch.export.export(model, (example_input,))

    # ── 4. ExecuTorch edge IR 변환 ────────────────────────────────────────────
    print("[3/5] to_edge() ...")
    from executorch.exir import to_edge, EdgeCompileConfig
    try:
        edge_prog = to_edge(exported)
    except Exception:
        # BatchNorm legit functional 예외 허용 후 재시도
        edge_prog = to_edge(
            exported,
            compile_config=EdgeCompileConfig(
                _core_aten_ops_exception_list=[
                    torch.ops.aten._native_batch_norm_legit_functional.default
                ]
            ),
        )

    # ── 5. 백엔드 위임 ────────────────────────────────────────────────────────
    if use_xnnpack:
        print("[4/5] XNNPACK 백엔드 적용 중...")
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            edge_prog = edge_prog.to_backend(XnnpackPartitioner())
            print("      XNNPACK 적용 완료.")
        except Exception as e:
            print(f"      Warning: XNNPACK 적용 실패 ({e}), 백엔드 없이 진행.")
    else:
        print("[4/5] 백엔드 위임 건너뜀.")

    # ── 6. 직렬화 ─────────────────────────────────────────────────────────────
    print("[5/5] .pte 저장 중...")
    exec_prog = edge_prog.to_executorch()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        exec_prog.write_to_file(f)

    size_kb = out_path.stat().st_size / 1024
    print(f"\n저장 완료: {out_path}  ({size_kb:.1f} KB)")

    # ── 7. 추론 검증 ──────────────────────────────────────────────────────────
    if verify:
        print("\n[검증] 내보낸 .pte 모델 추론 테스트...")
        try:
            from executorch.runtime import Runtime, Verification
            rt = Runtime.get()
            program = rt.load_program(str(out_path), verification=Verification.Minimal)
            method = program.load_method("forward")
            pte_output = method.execute([example_input])[0]
            diff = (ref_output - pte_output).abs().max().item()
            print(f"      최대 출력 오차: {diff:.6f}")
            if diff < 1e-3:
                print("      OK — 원본과 거의 동일.")
            else:
                print("      Warning: 오차가 큽니다. 양자화 영향일 수 있습니다.")
        except Exception as e:
            print(f"      검증 실패 (runtime 없음 또는 오류): {e}")

    return out_path


# =============================================================================
# 성능 평가
# =============================================================================

def _metrics_from_arrays(labels, preds, probs, class_names):
    """(N,C) numpy arrays → per-class + macro 지표 dict"""
    per_class = {}
    for i, name in enumerate(class_names):
        y_t = labels[:, i]
        y_p = preds[:, i]
        y_s = probs[:, i]
        if y_t.sum() == 0:
            per_class[name] = {k: float("nan") for k in
                               ["f1", "sens", "spec", "auroc", "auprc", "support"]}
            continue
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens  = tp / (tp + fn + 1e-12)
        spec  = tn / (tn + fp + 1e-12)
        f1    = f1_score(y_t, y_p, zero_division=0)
        try:    auroc = roc_auc_score(y_t, y_s)
        except: auroc = float("nan")
        try:    auprc = average_precision_score(y_t, y_s)
        except: auprc = float("nan")
        per_class[name] = {"f1": f1, "sens": sens, "spec": spec,
                           "auroc": auroc, "auprc": auprc, "support": int(y_t.sum())}

    def _nanmean(key):
        vals = [v[key] for v in per_class.values() if np.isfinite(v[key])]
        return float(np.mean(vals)) if vals else float("nan")

    macro = {k: _nanmean(k) for k in ["f1", "sens", "spec", "auroc", "auprc"]}
    return {"per_class": per_class, "macro": macro}


def _print_table(metrics, class_names, title):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")
    hdr = f"  {'Class':<14} {'F1':>6} {'Sens':>6} {'Spec':>6} {'AUROC':>6} {'AUPRC':>6} {'N':>7}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for name, row in metrics["per_class"].items():
        if not np.isfinite(row["support"]):
            continue
        f1    = f"{row['f1']:.3f}"    if np.isfinite(row['f1'])    else "  nan"
        sens  = f"{row['sens']:.3f}"  if np.isfinite(row['sens'])  else "  nan"
        spec  = f"{row['spec']:.3f}"  if np.isfinite(row['spec'])  else "  nan"
        auroc = f"{row['auroc']:.3f}" if np.isfinite(row['auroc']) else "  nan"
        auprc = f"{row['auprc']:.3f}" if np.isfinite(row['auprc']) else "  nan"
        print(f"  {name:<14} {f1:>6} {sens:>6} {spec:>6} {auroc:>6} {auprc:>6} {row['support']:>7,}")
    m = metrics["macro"]
    print("  " + "─" * (len(hdr) - 2))
    f1    = f"{m['f1']:.3f}"    if np.isfinite(m['f1'])    else "  nan"
    sens  = f"{m['sens']:.3f}"  if np.isfinite(m['sens'])  else "  nan"
    spec  = f"{m['spec']:.3f}"  if np.isfinite(m['spec'])  else "  nan"
    auroc = f"{m['auroc']:.3f}" if np.isfinite(m['auroc']) else "  nan"
    auprc = f"{m['auprc']:.3f}" if np.isfinite(m['auprc']) else "  nan"
    print(f"  {'macro':<14} {f1:>6} {sens:>6} {spec:>6} {auroc:>6} {auprc:>6}")
    print(f"{'─'*62}")


def _get_test_dataloader(split_json, class_names, batch_size=512, workers=4):
    from datamodule.HiCardiDataModule import HiCardiDataModule
    dm = HiCardiDataModule(
        split_json=split_json,
        return_arg={
            "waveform":    True,
            "labels":      True,
            "mode":        "beat",
            "n_classes":   len(class_names),
            "label_names": class_names,
        },
        undersampling_arg={"strategy": "none", "apply_to_train": False,
                           "apply_to_val": False, "apply_to_test": False},
        dataloader_arg={"batch_size": batch_size, "num_workers": workers,
                        "pin_memory": False, "drop_last": False},
    )
    dm.setup()
    return dm.bce_dataloader("test", batch_size, workers)


def evaluate_torch(model, dataloader, class_names, threshold=0.5):
    """PyTorch 모델로 test set 전체 추론 → 성능 지표"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            if x.ndim == 2:
                x = x.unsqueeze(1)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs > threshold).astype(int)
    return _metrics_from_arrays(labels, preds, probs, class_names)


def evaluate_pte(pte_path, dataloader, class_names, threshold=0.5, max_beats=None):
    """ExecuTorch .pte 모델로 test set 추론 → 성능 지표 (batch=1씩 처리)"""
    try:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer
        with open(pte_path, "rb") as f:
            et_module = _load_for_executorch_from_buffer(f.read())

        all_probs, all_labels = [], []
        n_total = max_beats or len(dataloader.dataset)
        n_done  = 0
        for x, y in dataloader:
            if x.ndim == 2:
                x = x.unsqueeze(1)
            for i in range(len(x)):
                if max_beats and n_done >= max_beats:
                    break
                xi = x[i:i+1]                          # (1,1,501) — batch=1 고정
                logits = et_module.forward([xi])[0]
                probs  = torch.sigmoid(logits).numpy()  # (1,7)
                all_probs.append(probs)
                all_labels.append(y[i:i+1].numpy())
                n_done += 1
                if n_done % 1000 == 0:
                    print(f"  {n_done:,} / {n_total:,} beats 완료", end="\r")
            if max_beats and n_done >= max_beats:
                break

        print(f"  {n_done:,} beats 평가 완료          ")
        probs  = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        preds  = (probs > threshold).astype(int)
        return _metrics_from_arrays(labels, preds, probs, class_names)

    except Exception as e:
        print(f"      [.pte 추론 실패] {e}")
        return None


# =============================================================================
# main
# =============================================================================

def _find_best_ckpt(exp_name: str = "demo_model") -> str:
    """results/{exp_name}/checkpoints/ 에서 가장 최근 best ckpt 자동 탐색."""
    ckpt_dir = config.RESULTS_DIR / exp_name / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_dir}")
    return str(ckpts[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",         type=str,   default=None,
                    help="체크포인트 경로 (.ckpt). 생략 시 demo_model에서 자동 탐색")
    ap.add_argument("--exp_name",     type=str,   default="demo_model",
                    help="--ckpt 생략 시 탐색할 실험 이름")
    ap.add_argument("--out",          type=str,   default=None,
                    help="출력 .pte 경로. 기본: results/{exp_name}/model.pte")
    ap.add_argument("--num_classes",  type=int,   default=len(DEFAULT_BEAT_CLASSES))
    ap.add_argument("--out_ch",       type=int,   default=config.MODEL_OUT_CH)
    ap.add_argument("--mid_ch",       type=int,   default=config.MODEL_MID_CH)
    ap.add_argument("--no_xnnpack",   action="store_true",
                    help="XNNPACK 백엔드 적용 안 함")
    ap.add_argument("--verify",       action="store_true",
                    help="내보낸 .pte 추론 결과를 원본과 비교")
    ap.add_argument("--workers",      type=int,   default=4,
                    help="DataLoader worker 수")
    ap.add_argument("--threshold",    type=float, default=0.5,
                    help="분류 임계값")
    ap.add_argument("--pte",          type=str,   default=None,
                    help="기존 .pte 경로 지정 시 export 생략하고 평가만 실행")
    ap.add_argument("--max_beats",    type=int,   default=10000,
                    help=".pte 평가 비트 수 상한 (0=전체, 기본 10000)")
    args = ap.parse_args()

    ckpt        = args.ckpt or _find_best_ckpt(args.exp_name)
    out         = args.out  or str(config.RESULTS_DIR / args.exp_name / "model.pte")
    class_names = list(DEFAULT_BEAT_CLASSES)
    max_beats   = args.max_beats if args.max_beats > 0 else None

    # ── 1. PyTorch 모델 로드 ──────────────────────────────────────────────────
    torch_model = load_model(ckpt, args.num_classes, args.out_ch, args.mid_ch)

    # ── 2. Test set 로드 ──────────────────────────────────────────────────────
    print("\n[Test set 로드 중...]")
    test_dl = _get_test_dataloader(
        split_json  = str(config.SPLIT_JSON),
        class_names = class_names,
        batch_size  = 512,
        workers     = args.workers,
    )
    print(f"  test beats: {len(test_dl.dataset):,}")

    # ── 3. Export 전 성능표 ───────────────────────────────────────────────────
    print("\n[Export 전 — PyTorch 모델 평가 중...]")
    before_metrics = evaluate_torch(torch_model, test_dl, class_names, args.threshold)
    _print_table(before_metrics, class_names, "Export 전 (PyTorch .ckpt)  — Test Set")

    # ── 4. ExecuTorch 변환 (--pte 지정 시 생략) ───────────────────────────────
    if args.pte:
        pte_path = args.pte
        print(f"\n[기존 .pte 사용] {pte_path}")
    else:
        pte_path = export_pte(
            ckpt_path   = ckpt,
            out_path    = out,
            num_classes = args.num_classes,
            out_ch      = args.out_ch,
            mid_ch      = args.mid_ch,
            use_xnnpack = not args.no_xnnpack,
            verify      = args.verify,
        )

    # ── 5. Export 후 성능표 ───────────────────────────────────────────────────
    limit_str = f"첫 {max_beats:,}비트" if max_beats else "전체"
    print(f"\n[Export 후 — ExecuTorch .pte 평가 중... ({limit_str})]")
    after_metrics = evaluate_pte(str(pte_path), test_dl, class_names, args.threshold, max_beats)
    if after_metrics:
        _print_table(after_metrics, class_names, f"Export 후 (ExecuTorch .pte) — Test Set ({limit_str})")


if __name__ == "__main__":
    main()
