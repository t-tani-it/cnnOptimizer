# Why: 全設定値集約で再利用性確保 / What: 探索範囲・学習既定値 / Assumption: Optuna 4系 / I-O: 定数参照 / Caution: Test混用禁止 / Future: yaml切出 / Change Log: 2026-09-05 new
"""共通設定モジュール。

旧コード固定値を範囲化した経緯は docs/old_vs_modern.md 参照。
"""

from __future__ import annotations

# 解説: 旧seed=2とval10%を継承し比較可能性を保つ
RANDOM_SEED: int = 2
VAL_SIZE: float = 0.1
MAX_EPOCHS: int = 40
EARLY_PATIENCE: int = 5
LR_PATIENCE: int = 4
LR_FACTOR: float = 0.2
MIN_LR: float = 1e-4
N_TRIALS_DEFAULT: int = 50
OPTUNA_DB: str = "results/optuna.db"
RESULTS_DB: str = "results/results.db"

# 解説: 旧dom1=64, dom2=400-700, do=0.25-0.3等を範囲化。固定は精度を下げないが最適値を逃すため
SEARCH_SPACE: dict = {
    "conv1_filters": (32, 128),
    "conv2_filters": (128, 512),
    "n_conv_blocks": (2, 3),
    "kernel_size": [3, 5],
    "dense_units": (128, 1024),
    "dropout": (0.1, 0.5),
    "activation": ["relu", "mish", "tanh"],
    "optimizer": ["adam", "adamw"],
    "lr": (1e-4, 1e-2),
    "batch_size": [32, 64, 128, 256],
    "rotation": (0.0, 20.0),
    "shift": (0.0, 0.2),
    "zoom": (0.0, 0.2),
}
