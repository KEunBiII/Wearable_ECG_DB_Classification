# -*- coding: utf-8 -*-
"""
mat_viewer.py  —  HiCardi .mat ECG 뷰어 (GUI)

실행:
    python validation/mat_viewer.py
    python validation/mat_viewer.py --root /path/to/mezoo_db

레이아웃:
    왼쪽  : 코호트 리스트 → 레코드 리스트 → 확인 버튼
    오른쪽: 분석 정보 패널 + ECG 스트립 뷰 (슬라이더 / 이전·다음 버튼)
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np

try:
    import h5py
    _H5PY_OK = True
except ImportError:
    _H5PY_OK = False

# ── 경로 ──────────────────────────────────────────────────────────────────────
_REPO_ROOT   = Path(__file__).resolve().parent.parent
MEZOO_ROOT   = _REPO_ROOT / "Database" / "mezoo_db"

# ── 라벨 스키마 ───────────────────────────────────────────────────────────────
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


LABEL_COLORS: dict[str, str] = {
    "lead_off":    "#808080",
    "Normal":      "#44cc66",
    "VF/VT":       "#cc2222",
    "VPC":         "#ff5555",
    "Bigeminy":    "#ff88cc",
    "Trigeminy":   "#ffaa88",
    "Bradycardia": "#4488ff",
    "AF/AFL":      "#aa44ff",
    "APC":         "#ff9900",
    "Sinus Tachy": "#ffdd00",
}
LABEL_ALPHA    = 0.28
LEADOFF_MIN_DUR = 1.0


# =============================================================================
# 1) 데이터 레이어
# =============================================================================

@dataclass
class HolterRecord:
    mat_path:   str
    ecg_mv:     np.ndarray                    # (N,) float32  raw mV
    fs:         float
    rpeak_idx:  Optional[np.ndarray] = None   # R-peak sample indices
    lead_off:   Optional[np.ndarray] = None   # (N,) int8
    final_flag: Optional[np.ndarray] = None   # (N, 27) int8
    mat_keys:   list[str] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return len(self.ecg_mv) / max(self.fs, 1.0)

    @property
    def n_beats(self) -> int:
        return 0 if self.rpeak_idx is None else int(len(self.rpeak_idx))

    @property
    def lead_off_sec(self) -> float:
        if self.lead_off is None:
            return 0.0
        return float((self.lead_off > 0).sum()) / max(self.fs, 1.0)

    @property
    def valid_duration_sec(self) -> float:
        return max(self.duration_sec - self.lead_off_sec, 0.0)


class MatLoader:
    """HDF5 v7.3 .mat 파일을 raw signal로 로드 (필터 없음, 실패 필드는 패스)."""

    @staticmethod
    def _as_1d(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr).reshape(-1)

    @classmethod
    def load(cls, mat_path: str) -> HolterRecord:
        if not _H5PY_OK:
            raise RuntimeError("h5py가 설치되어 있지 않습니다: pip install h5py")
        if not Path(mat_path).exists():
            raise FileNotFoundError(mat_path)

        with h5py.File(mat_path, "r") as f:
            keys = list(f.keys())

            def get(name: str):
                if name in f:
                    return f[name]
                for k in keys:
                    if k.lower() == name.lower():
                        return f[k]
                raise KeyError(name)

            # ECG (필수)
            raw = cls._as_1d(np.array(get("dECG")))
            adc = np.nan_to_num(raw, nan=8192.0, posinf=8192.0, neginf=8192.0).astype(np.int32)
            ecg_mv = adc .astype(np.float32)

            # 샘플링 주파수
            fs = 250.0
            for k in ["fs", "Fs", "FS"]:
                try:
                    fs = float(cls._as_1d(np.array(get(k)))[0])
                    break
                except KeyError:
                    pass

            # R-peak (선택)
            rpeak_idx = None
            for k in ["Rpk_label", "Rpk_flag", "RpkLabel", "RpkFlag"]:
                try:
                    rpk = cls._as_1d(np.array(get(k)))
                    rpeak_idx = np.where(np.nan_to_num(rpk) > 0)[0].astype(np.int64)
                    break
                except KeyError:
                    pass

            # Lead-off (선택)
            lead_off = None
            for k in ["LeadOff", "lead_off", "leadoff"]:
                try:
                    lo = cls._as_1d(np.array(get(k)))
                    lead_off = np.nan_to_num(lo, nan=1.0).astype(np.int8)
                    break
                except KeyError:
                    pass

            # final_flag 라벨 (선택, HDF5에서 (27,N) → (N,27))
            final_flag = None
            try:
                ff = np.array(get("final_flag"))
                if ff.ndim == 2:
                    if ff.shape[0] == 27 and ff.shape[1] != 27:
                        ff = ff.T
                    if ff.shape[1] == 27 and ff.shape[0] == len(ecg_mv):
                        final_flag = np.where(ff > 0, 1, 0).astype(np.int8)
            except Exception:
                pass

        return HolterRecord(
            mat_path=mat_path, ecg_mv=ecg_mv, fs=fs,
            rpeak_idx=rpeak_idx, lead_off=lead_off,
            final_flag=final_flag, mat_keys=keys,
        )


# =============================================================================
# 2) 분석 레이어
# =============================================================================

class RecordAnalyzer:
    """레코드 메타 정보 추출 + 오버레이용 구간 계산."""

    @staticmethod
    def summarize(rec: HolterRecord) -> dict[str, str]:
        info: dict[str, str] = {}
        info["fs (Hz)"]    = f"{rec.fs:.1f}"
        info["Samples"]    = f"{len(rec.ecg_mv):,}"
        info["Duration"]   = f"{rec.duration_sec:.1f} s  ({rec.duration_sec / 3600:.2f} h)"
        info["R-peaks"]    = f"{rec.n_beats:,}" if rec.rpeak_idx is not None else "N/A"
        info["Lead-off"]   = f"{rec.lead_off_sec:.1f} s" if rec.lead_off is not None else "N/A"
        info["final_flag"] = "있음" if rec.final_flag is not None else "없음"
        if rec.n_beats > 0 and rec.valid_duration_sec > 0:
            info["Mean HR"] = f"{rec.n_beats / rec.valid_duration_sec * 60:.1f} bpm"
        if rec.final_flag is not None:
            active = [
                name for name, col in HICARDI_LABEL_COL_MAP.items()
                if col < rec.final_flag.shape[1] and rec.final_flag[:, col].any()
            ]
            info["Labels (active)"] = ", ".join(active) if active else "없음"
        info["MAT keys"] = ", ".join(rec.mat_keys)
        return info

    @staticmethod
    def lead_off_intervals(rec: HolterRecord,
                           min_dur: float = LEADOFF_MIN_DUR) -> list[tuple[float, float]]:
        if rec.lead_off is None:
            return []
        mask = rec.lead_off > 0
        ivs: list[tuple[float, float]] = []
        in_z, start = False, 0
        for i, z in enumerate(mask):
            if z and not in_z:
                in_z, start = True, i
            elif not z and in_z:
                in_z = False
                if (i - start) / rec.fs >= min_dur:
                    ivs.append((start / rec.fs, i / rec.fs))
        if in_z and (len(mask) - start) / rec.fs >= min_dur:
            ivs.append((start / rec.fs, len(mask) / rec.fs))
        return ivs

    @staticmethod
    def label_intervals(rec: HolterRecord) -> dict[str, list[tuple[float, float]]]:
        """final_flag (N, 27) → 라벨별 시간 구간 (sample-level, 연속 구간 병합)."""
        result: dict[str, list[tuple[float, float]]] = {}
        if rec.final_flag is None:
            return result
        fs = rec.fs
        for name, col in HICARDI_LABEL_COL_MAP.items():
            if col >= rec.final_flag.shape[1]:
                continue
            idxs = np.where(rec.final_flag[:, col] > 0)[0]
            if len(idxs) == 0:
                continue
            ivs: list[tuple[float, float]] = []
            s = e = int(idxs[0])
            gap = int(fs)           # 1초 이내 gap은 병합
            for idx in idxs[1:]:
                idx = int(idx)
                if idx - e <= gap:
                    e = idx
                else:
                    ivs.append((s / fs, e / fs))
                    s = e = idx
            ivs.append((s / fs, e / fs))
            result[name] = ivs
        return result


# =============================================================================
# 3) NeuroKit2 R-peak 재검출
# =============================================================================

class NeuroKitDetector:
    """NeuroKit2로 ECG 구간의 R-peak를 재검출 (설치 없으면 graceful skip)."""

    _available: Optional[bool] = None

    @classmethod
    def is_available(cls) -> bool:
        if cls._available is None:
            try:
                import neurokit2  # noqa: F401
                cls._available = True
            except ImportError:
                cls._available = False
        return cls._available

    @classmethod
    def detect_window(cls, ecg: np.ndarray, fs: float,
                      s0: int, s1: int,
                      context_sec: float = 5.0) -> np.ndarray:
        """
        전체 신호 ecg 에서 [s0, s1) 구간의 NK2 R-peak 인덱스를 반환.
        앞뒤 context_sec 만큼 여유를 두고 검출한 뒤 범위 내 peak만 추림.
        """
        if not cls.is_available():
            return np.empty(0, dtype=np.int64)
        import neurokit2 as nk

        ctx   = int(context_sec * fs)
        start = max(0, s0 - ctx)
        end   = min(len(ecg), s1 + ctx)
        chunk = ecg[start:end].astype(np.float64)

        # 센터링 (ADC offset 제거)
        chunk = chunk - np.median(chunk)

        try:
            _, info = nk.ecg_peaks(chunk, sampling_rate=int(fs), method="neurokit")
            local_peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
        except Exception:
            return np.empty(0, dtype=np.int64)

        global_peaks = local_peaks + start
        return global_peaks[(global_peaks >= s0) & (global_peaks < s1)]


# =============================================================================
# 4) 플롯 레이어
# =============================================================================

class StripPlotter:
    """matplotlib Figure에 ECG 스트립 행을 그리는 클래스."""

    ROWS_PER_VIEW   = 5
    SECONDS_PER_ROW = 10.0

    def __init__(self, fig: plt.Figure):
        self._fig = fig

    def draw(self, rec: HolterRecord, t_start: float,
             analyzer: RecordAnalyzer,
             show_labels:  bool = True,
             show_rpeaks:  bool = True,
             nk2_rpeaks:   Optional[np.ndarray] = None):
        self._fig.clear()

        n_rows = self.ROWS_PER_VIEW
        spr    = self.SECONDS_PER_ROW
        fs     = rec.fs
        sig    = rec.ecg_mv

        lo_ivs  = analyzer.lead_off_intervals(rec)
        lbl_ivs = analyzer.label_intervals(rec) if show_labels else {}

        axes = self._fig.subplots(n_rows, 1, squeeze=False)

        for r in range(n_rows):
            ax = axes[r, 0]
            t0 = t_start + r * spr
            t1 = t0 + spr
            s0 = int(t0 * fs)
            s1 = min(int(t1 * fs), len(sig))

            if s0 >= len(sig):
                ax.set_visible(False)
                continue

            chunk = sig[s0:s1]
            n_pad = int((t1 - t0) * fs) - len(chunk)
            if n_pad > 0:
                chunk = np.pad(chunk, (0, n_pad))

            t_arr = np.linspace(t0, t1, len(chunk))

            ymin, ymax = float(chunk.min()), float(chunk.max())
            pad = max((ymax - ymin) * 0.15, 0.3)
            ax.set_xlim(t0, t1)
            ax.set_ylim(ymin - pad, ymax + pad)

            # lead-off 오버레이
            for ts, te in lo_ivs:
                lo, hi = max(ts, t0), min(te, t1)
                if lo < hi:
                    ax.axvspan(lo, hi, color=LABEL_COLORS["lead_off"],
                               alpha=LABEL_ALPHA, linewidth=0, zorder=1)

            # 라벨 오버레이
            for name, ivs in lbl_ivs.items():
                color = LABEL_COLORS.get(name, "#aaaaaa")
                for ts, te in ivs:
                    lo, hi = max(ts, t0), min(te, t1)
                    if lo < hi:
                        ax.axvspan(lo, hi, color=color, alpha=LABEL_ALPHA,
                                   linewidth=0, zorder=1)

            ax.plot(t_arr, chunk, linewidth=0.6, color="#1a1a1a", zorder=2)

            # 원본 R-peak 마커 (빨강)
            if show_rpeaks and rec.rpeak_idx is not None:
                r_in = rec.rpeak_idx[(rec.rpeak_idx >= s0) & (rec.rpeak_idx < s1)]
                if len(r_in):
                    ax.vlines(r_in / fs, ymin - pad * 0.2, ymax + pad * 0.2,
                              color="#cc3333", linewidth=0.7, alpha=0.75, zorder=3)

            # NeuroKit2 R-peak 마커 (청록)
            if nk2_rpeaks is not None:
                nk_in = nk2_rpeaks[(nk2_rpeaks >= s0) & (nk2_rpeaks < s1)]
                if len(nk_in):
                    ax.vlines(nk_in / fs, ymin - pad * 0.35, ymax + pad * 0.35,
                              color="#00bbaa", linewidth=0.9, alpha=0.85,
                              linestyle="--", zorder=4)

            ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
            ax.grid(which="major", axis="x", color="#e8b0b0",
                    linewidth=0.5, alpha=0.8, zorder=0)
            ax.grid(which="minor", axis="x", color="#f5d5d5",
                    linewidth=0.25, alpha=0.6, zorder=0)
            ax.grid(which="major", axis="y", color="#e8b0b0",
                    linewidth=0.3, alpha=0.5, zorder=0)
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{x:.0f}s"))
            ax.tick_params(axis="x", labelsize=6, length=2, pad=1)
            ax.tick_params(axis="y", labelsize=5, length=2, pad=1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.5)
            ax.spines["bottom"].set_linewidth(0.5)

        # 범례 (라벨 모드일 때, 첫 번째 페이지처럼 figure 상단에)
        if show_labels and lbl_ivs:
            patches = [
                mpatches.Patch(facecolor=LABEL_COLORS.get(n, "#aaa"),
                               alpha=0.8, edgecolor="none", label=n)
                for n in lbl_ivs
            ]
            if lo_ivs:
                patches.insert(0, mpatches.Patch(
                    facecolor=LABEL_COLORS["lead_off"],
                    alpha=0.8, edgecolor="none", label="lead_off"))
            self._fig.legend(handles=patches, loc="upper right",
                             ncol=min(len(patches), 5),
                             fontsize=6, frameon=False,
                             handlelength=1.2, handleheight=0.8,
                             columnspacing=0.8, handletextpad=0.4)

        self._fig.tight_layout(pad=0.5, h_pad=0.3)
        self._fig.canvas.draw_idle()


# =============================================================================
# 4) GUI 레이어
# =============================================================================

class SelectorPanel(ttk.Frame):
    """왼쪽: 코호트 리스트 → 레코드 리스트 → 확인 버튼."""

    def __init__(self, parent, db_root: Path, on_confirm):
        super().__init__(parent, width=230)
        self.pack_propagate(False)
        self._db_root    = db_root
        self._on_confirm = on_confirm
        self._cohorts: list[Path] = []
        self._records: list[Path] = []
        self._build()
        self._populate_cohorts()

    # ── 빌드 ─────────────────────────────────────────────────────────────────
    def _build(self):
        ttk.Label(self, text="코호트", font=("", 9, "bold")).pack(
            anchor="w", padx=6, pady=(8, 2))
        f1 = ttk.Frame(self)
        f1.pack(fill="both", expand=True, padx=6)
        sb1 = ttk.Scrollbar(f1, orient="vertical")
        self._cohort_lb = tk.Listbox(f1, yscrollcommand=sb1.set,
                                     selectmode="single", height=8,
                                     exportselection=False, font=("", 8))
        sb1.config(command=self._cohort_lb.yview)
        self._cohort_lb.pack(side="left", fill="both", expand=True)
        sb1.pack(side="right", fill="y")
        self._cohort_lb.bind("<<ListboxSelect>>", self._on_cohort_select)

        ttk.Label(self, text="레코드", font=("", 9, "bold")).pack(
            anchor="w", padx=6, pady=(8, 2))
        f2 = ttk.Frame(self)
        f2.pack(fill="both", expand=True, padx=6)
        sb2 = ttk.Scrollbar(f2, orient="vertical")
        self._rec_lb = tk.Listbox(f2, yscrollcommand=sb2.set,
                                   selectmode="single", height=14,
                                   exportselection=False, font=("", 8))
        sb2.config(command=self._rec_lb.yview)
        self._rec_lb.pack(side="left", fill="both", expand=True)
        sb2.pack(side="right", fill="y")

        bf = ttk.Frame(self)
        bf.pack(fill="x", padx=6, pady=8)
        self._btn = ttk.Button(bf, text="확인", command=self._confirm)
        self._btn.pack(fill="x")

        self._status = ttk.Label(self, text="", font=("", 7),
                                  foreground="#555555", wraplength=210)
        self._status.pack(padx=6, pady=(0, 6))

    # ── 코호트 목록 채우기 ────────────────────────────────────────────────────
    def _populate_cohorts(self):
        if not self._db_root.exists():
            self._status.config(text=f"경로 없음:\n{self._db_root}")
            return
        self._cohorts = sorted(d for d in self._db_root.iterdir() if d.is_dir())
        self._cohort_lb.delete(0, "end")
        for d in self._cohorts:
            self._cohort_lb.insert("end", d.name)
        if self._cohorts:
            self._cohort_lb.selection_set(0)
            self._on_cohort_select(None)

    def _on_cohort_select(self, _):
        sel = self._cohort_lb.curselection()
        if not sel:
            return
        mats = sorted(self._cohorts[sel[0]].glob("*.mat"))
        self._records = mats
        self._rec_lb.delete(0, "end")
        for m in mats:
            self._rec_lb.insert("end", m.stem)
        if mats:
            self._rec_lb.selection_set(0)

    def _confirm(self):
        r_sel = self._rec_lb.curselection()
        if not r_sel:
            messagebox.showwarning("선택 없음", "레코드를 선택하세요.")
            return
        mat_path = str(self._records[r_sel[0]])
        self._status.config(text=f"로딩 중…")
        self._btn.config(state="disabled")
        self._on_confirm(mat_path, self._done_cb)

    def _done_cb(self, error: str | None):
        self._btn.config(state="normal")
        self._status.config(
            text="로딩 완료" if error is None else f"오류: {error}")


class InfoPanel(ttk.Frame):
    """분석 정보 표시 패널."""

    def __init__(self, parent):
        super().__init__(parent)
        ttk.Label(self, text="레코드 분석 정보", font=("", 9, "bold")).pack(
            anchor="w", padx=8, pady=(4, 2))
        self._grid = ttk.Frame(self)
        self._grid.pack(fill="x", padx=8, pady=(0, 4))

    def update(self, info: dict[str, str]):
        for w in self._grid.winfo_children():
            w.destroy()
        for r, (k, v) in enumerate(info.items()):
            ttk.Label(self._grid, text=k + ":", font=("", 7, "bold")).grid(
                row=r, column=0, sticky="w", padx=(0, 6))
            ttk.Label(self._grid, text=v, font=("", 7)).grid(
                row=r, column=1, sticky="w")

    def clear(self):
        for w in self._grid.winfo_children():
            w.destroy()


class ECGCanvas(ttk.Frame):
    """오른쪽: matplotlib 스트립 뷰 + 탐색 컨트롤."""

    def __init__(self, parent):
        super().__init__(parent)
        self._rec: Optional[HolterRecord] = None
        self._analyzer  = RecordAnalyzer()
        self._t_start   = 0.0

        # NK2 관련 상태
        self._nk2_computing = False     # 백그라운드 스레드 실행 중 여부

        self._fig = plt.figure(figsize=(10, 6))
        self._plotter = StripPlotter(self._fig)

        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        # ── 탐색 컨트롤 ──────────────────────────────────────────────────────
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", pady=2)

        ttk.Button(ctrl, text="◀◀", width=3,
                   command=self._go_start).pack(side="left", padx=2)
        ttk.Button(ctrl, text="◀",  width=3,
                   command=self._go_back).pack(side="left", padx=2)
        self._slider = ttk.Scale(ctrl, orient="horizontal", from_=0, to=1,
                                  command=self._on_slider)
        self._slider.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(ctrl, text="▶",  width=3,
                   command=self._go_forward).pack(side="right", padx=2)
        ttk.Button(ctrl, text="▶▶", width=3,
                   command=self._go_end).pack(side="right", padx=2)

        self._time_lbl = ttk.Label(ctrl, text="", font=("", 8), width=22)
        self._time_lbl.pack(side="right", padx=6)

        # ── 표시 옵션 ────────────────────────────────────────────────────────
        opt = ttk.Frame(self)
        opt.pack(fill="x", padx=4, pady=(0, 2))
        self._show_labels = tk.BooleanVar(value=True)
        self._show_rpeaks = tk.BooleanVar(value=True)
        self._show_nk2    = tk.BooleanVar(value=False)

        ttk.Checkbutton(opt, text="라벨 오버레이",
                        variable=self._show_labels,
                        command=self._redraw).pack(side="left", padx=4)
        ttk.Checkbutton(opt, text="원본 R-peak",
                        variable=self._show_rpeaks,
                        command=self._redraw).pack(side="left", padx=4)

        nk2_lbl = "(NK2 없음)" if not NeuroKitDetector.is_available() else ""
        self._nk2_cb = ttk.Checkbutton(
            opt, text=f"NeuroKit2 R-peak {nk2_lbl}",
            variable=self._show_nk2,
            command=self._on_nk2_toggle,
            state="normal" if NeuroKitDetector.is_available() else "disabled",
        )
        self._nk2_cb.pack(side="left", padx=4)

        self._nk2_status = ttk.Label(opt, text="", font=("", 7),
                                      foreground="#007799")
        self._nk2_status.pack(side="left", padx=4)

    # ── 공개 메서드 ───────────────────────────────────────────────────────────
    def load(self, rec: HolterRecord):
        self._rec         = rec
        self._t_start     = 0.0
        self._nk2_computing = False
        self._show_nk2.set(False)
        self._nk2_status.config(text="")
        view_dur = StripPlotter.ROWS_PER_VIEW * StripPlotter.SECONDS_PER_ROW
        max_t    = max(rec.duration_sec - view_dur, 0.0)
        self._slider.config(from_=0, to=max(max_t, 1.0))
        self._slider.set(0.0)
        self._redraw()

    def clear(self):
        self._rec           = None
        self._nk2_computing = False
        self._show_nk2.set(False)
        self._nk2_status.config(text="")
        self._fig.clear()
        self._canvas.draw_idle()
        self._time_lbl.config(text="")

    # ── 내부 ─────────────────────────────────────────────────────────────────
    def _on_nk2_toggle(self):
        if not self._show_nk2.get():
            self._nk2_status.config(text="")
            self._redraw()
            return
        self._trigger_nk2()

    def _trigger_nk2(self):
        """현재 뷰 구간을 백그라운드 스레드에서 NK2 검출 후 redraw."""
        if self._rec is None or self._nk2_computing:
            return
        rec      = self._rec
        t_start  = self._t_start
        fs       = rec.fs
        spr      = StripPlotter.SECONDS_PER_ROW
        n_rows   = StripPlotter.ROWS_PER_VIEW
        s0       = int(t_start * fs)
        s1       = min(int((t_start + n_rows * spr) * fs), len(rec.ecg_mv))

        self._nk2_computing = True
        self._nk2_status.config(text="NK2 계산 중…")

        def _worker():
            peaks = NeuroKitDetector.detect_window(rec.ecg_mv, fs, s0, s1)
            self.after(0, lambda: self._nk2_done(rec, peaks))

        threading.Thread(target=_worker, daemon=True).start()

    def _nk2_done(self, rec: HolterRecord, peaks: np.ndarray):
        self._nk2_computing = False
        if self._rec is not rec:   # 레코드가 바뀐 경우 무시
            return
        n = len(peaks)
        self._nk2_status.config(text=f"NK2: {n}개")
        self._redraw(nk2_override=peaks)

    def _redraw(self, nk2_override: Optional[np.ndarray] = None):
        if self._rec is None:
            return

        nk2 = nk2_override if (self._show_nk2.get() and nk2_override is not None) else None

        self._plotter.draw(self._rec, self._t_start, self._analyzer,
                           show_labels=self._show_labels.get(),
                           show_rpeaks=self._show_rpeaks.get(),
                           nk2_rpeaks=nk2)
        view_dur = StripPlotter.ROWS_PER_VIEW * StripPlotter.SECONDS_PER_ROW
        t_end    = min(self._t_start + view_dur, self._rec.duration_sec)
        self._time_lbl.config(
            text=f"{self._t_start:.0f}s – {t_end:.0f}s / {self._rec.duration_sec:.0f}s")

    def _navigate(self, new_t: float):
        """공통 탐색: t_start 갱신 → redraw → NK2 재계산."""
        self._t_start = new_t
        self._redraw()
        if self._show_nk2.get():
            self._trigger_nk2()

    def _on_slider(self, val):
        self._navigate(float(val))

    def _go_back(self):
        step = StripPlotter.SECONDS_PER_ROW
        self._navigate(max(0.0, self._t_start - step))
        self._slider.set(self._t_start)

    def _go_forward(self):
        if self._rec is None:
            return
        step  = StripPlotter.SECONDS_PER_ROW
        view  = StripPlotter.ROWS_PER_VIEW * step
        max_t = max(self._rec.duration_sec - view, 0.0)
        self._navigate(min(max_t, self._t_start + step))
        self._slider.set(self._t_start)

    def _go_start(self):
        self._navigate(0.0)
        self._slider.set(0.0)

    def _go_end(self):
        if self._rec is None:
            return
        view  = StripPlotter.ROWS_PER_VIEW * StripPlotter.SECONDS_PER_ROW
        max_t = max(self._rec.duration_sec - view, 0.0)
        self._navigate(max_t)
        self._slider.set(max_t)


# =============================================================================
# 5) 메인 앱
# =============================================================================

class MatViewerApp(tk.Tk):
    """메인 윈도우: 셀렉터 + 정보 패널 + ECG 캔버스."""

    def __init__(self, db_root: Path):
        super().__init__()
        self.title("HiCardi ECG Viewer")
        self.geometry("1300x820")
        self._current_rec: Optional[HolterRecord] = None
        self._build(db_root)

    def _build(self, db_root: Path):
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # 왼쪽: 선택 패널
        self._selector = SelectorPanel(paned, db_root, self._on_confirm)
        paned.add(self._selector, weight=0)

        # 오른쪽: 정보 + 캔버스
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._info = InfoPanel(right)
        self._info.pack(fill="x")

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=2)

        self._ecg = ECGCanvas(right)
        self._ecg.pack(fill="both", expand=True)

    def _on_confirm(self, mat_path: str, done_cb):
        # 이전 캐시 즉시 해제
        self._current_rec = None
        self._ecg.clear()
        self._info.clear()

        def _worker():
            try:
                rec = MatLoader.load(mat_path)
                self.after(0, lambda: self._load_done(rec, done_cb))
            except Exception as e:
                self.after(0, lambda: done_cb(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _load_done(self, rec: HolterRecord, done_cb):
        self._current_rec = rec
        self._info.update(RecordAnalyzer.summarize(rec))
        self._ecg.load(rec)
        done_cb(None)


# =============================================================================
# 진입점
# =============================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="HiCardi .mat ECG Viewer")
    ap.add_argument("--root", default=str(MEZOO_ROOT),
                    help=f"mezoo_db 루트 디렉터리 (기본: {MEZOO_ROOT})")
    args = ap.parse_args()

    db_root = Path(args.root)
    if not db_root.exists():
        print(f"[ERROR] 경로 없음: {db_root}")
        sys.exit(1)

    app = MatViewerApp(db_root)
    app.mainloop()


if __name__ == "__main__":
    main()
