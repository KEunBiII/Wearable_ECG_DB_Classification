# -*- coding: utf-8 -*-
"""
config.py — ECG_Classification_HiCardi 프로젝트 설정

라벨 클래스는 나중에 결정 후 LABEL_MAP에 채워넣으면 됩니다.
"""

import os
from pathlib import Path

# =============================================================================
# 경로 설정
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# Raw HiCardi .mat 파일 루트 (실제 경로로 수정)
HICARDI_ROOT = Path(os.environ.get("HICARDI_ROOT", PROJECT_ROOT / "Database" / "hicardi"))

# 비트 캐시 루트 (Hicardi_dataset_generation.py로 생성)
CACHE_ROOT   = Path(os.environ.get(
    "HICARDI_CACHE",
    PROJECT_ROOT / "hicardi_beat_cache_01holter"
))

# 데이터 스플릿 및 인덱스
SPLIT_JSON   = Path(os.environ.get("HICARDI_SPLIT", PROJECT_ROOT / "data" / "split_01holter.json"))
INDEX_CSV    = Path(os.environ.get("HICARDI_INDEX", PROJECT_ROOT / "hicardi_beat_cache_01holter" / "index.csv"))

# 결과 저장 경로
RESULTS_DIR  = PROJECT_ROOT / "results"

# =============================================================================
# HiCardi final_flag 컬럼 매핑 (27컬럼 multi-hot → 라벨 이름)
# =============================================================================
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

# =============================================================================
# 사용할 분류 클래스 (나중에 결정 후 여기에 입력)
# 예시) LABEL_MAP = {"Normal": 0, "VPC": 1, "AF/AFL": 2}
# 값은 최종 클래스 인덱스 (0부터 연속), 키는 HICARDI_LABEL_COL_MAP의 키와 동일
# =============================================================================
LABEL_MAP: dict[str, int] = {}   # ← 추후 결정

# 우선순위 (multi-hot beat에서 single class로 변환 시 이 순서로 선택)
# LABEL_MAP 결정 후 함께 수정
LABEL_PRIORITY: list[str] = []   # ← 추후 결정

# =============================================================================
# 신호 파라미터
# =============================================================================
WIN_LEN   = 501          # pre(250) + center(1) + post(250)
PRE       = 250
POST      = 250
TARGET_FS = 250.0        # HiCardi 기본 샘플링 주파수

# =============================================================================
# 모델 파라미터 (ResUNet)
# =============================================================================
MODEL_IN_CHANNELS = 1
MODEL_OUT_CH      = 180
MODEL_MID_CH      = 30

# =============================================================================
# 학습 파라미터 (추후 트레이너 결정 후 채움)
# =============================================================================
BATCH_SIZE   = 256
NUM_WORKERS  = 4
LR           = 1e-3
MAX_EPOCHS   = 50
SEED         = 42
