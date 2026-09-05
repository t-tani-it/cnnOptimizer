# Why: Optuna探索の中核 / What: suggest・学習・報告・保存 / Assumption: GPU自動 / I-O: trial->val_acc / Caution: Test不使用 / Future: なし / Change Log: 2026-09-05 new; 2026-09-06 fix optuna-integration分離対応
"""Keras用Optuna objective。"""

from __future__ import annotations

import time

import keras
import optuna

try:
    from optuna_integration.tfkeras import TFKerasPruningCallback
except ImportError:
    try:
        from optuna_integration import TFKerasPruningCallback
    except ImportError:
        from optuna.integration import TFKerasPruningCallback

from common.config import (
    EARLY_PATIENCE,
    LR_FACTOR,
    LR_PATIENCE,
    MAX_EPOCHS,
    MIN_LR,
    RESULTS_DB,
    SEARCH_SPACE,
)
from common.data import load_mnist_numpy
from common.results_db import save_trial

from .model import build_model

_DATA: tuple | None = None


def suggest_params(trial: optuna.Trial) -> dict:
    """探索空間提示。

    Args:
        trial: Optuna trial。

    Returns:
        params辞書。
    """
    # 解説: 旧固定値を範囲化。全組合わせでなくTPEが選択する
    return {
        "conv1_filters": trial.suggest_int(
            "conv1_filters", *SEARCH_SPACE["conv1_filters"], step=32
        ),
        "conv2_filters": trial.suggest_int(
            "conv2_filters", *SEARCH_SPACE["conv2_filters"], step=64
        ),
        "n_conv_blocks": trial.suggest_int(
            "n_conv_blocks", *SEARCH_SPACE["n_conv_blocks"]
        ),
        "kernel_size": trial.suggest_categorical(
            "kernel_size", SEARCH_SPACE["kernel_size"]
        ),
        "dense_units": trial.suggest_int(
            "dense_units", *SEARCH_SPACE["dense_units"], step=128
        ),
        "dropout": trial.suggest_float("dropout", *SEARCH_SPACE["dropout"]),
        "activation": trial.suggest_categorical(
            "activation", SEARCH_SPACE["activation"]
        ),
        "optimizer": trial.suggest_categorical("optimizer", SEARCH_SPACE["optimizer"]),
        "lr": trial.suggest_float("lr", *SEARCH_SPACE["lr"], log=True),
        "batch_size": trial.suggest_categorical(
            "batch_size", SEARCH_SPACE["batch_size"]
        ),
        "rotation": trial.suggest_float("rotation", *SEARCH_SPACE["rotation"]),
        "shift": trial.suggest_float("shift", *SEARCH_SPACE["shift"]),
        "zoom": trial.suggest_float("zoom", *SEARCH_SPACE["zoom"]),
    }


def validate_input(trial: optuna.Trial) -> None:
    """trial検査(予約)。"""
    if trial is None:
        raise ValueError("no trial")


def execute_logic(trial: optuna.Trial) -> float:
    """1 Trial学習。

    Args:
        trial: Optuna trial。

    Returns:
        val accuracy。
    """
    global _DATA
    if _DATA is None:
        _DATA = load_mnist_numpy()
    xtr, xva, _, ytr, yva, _ = _DATA
    params = suggest_params(trial)
    model = build_model(params)
    t0 = time.time()
    # 解説: 旧ES+ReduceLR継承、Pruningで不良Trial早期終了
    cbs = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR
        ),
        TFKerasPruningCallback(trial, "val_accuracy"),
    ]
    hist = model.fit(
        xtr,
        ytr,
        validation_data=(xva, yva),
        epochs=MAX_EPOCHS,
        batch_size=params["batch_size"],
        verbose=0,
        callbacks=cbs,
    )
    i = int(__import__("numpy").argmax(hist.history["val_accuracy"]))
    vacc, vloss = (
        float(hist.history["val_accuracy"][i]),
        float(hist.history["val_loss"][i]),
    )
    save_trial(
        RESULTS_DB,
        {
            "framework": "keras",
            "params": params,
            "val_acc": vacc,
            "val_loss": vloss,
            "elapsed": time.time() - t0,
        },
    )
    model.save(f"models/keras_trial{trial.number}_{vacc:.4f}.keras")
    return vacc


def format_output(score: float) -> float:
    """最大化用そのまま返却。"""
    return score


def objective(trial: optuna.Trial) -> float:
    """Optuna登録関数。"""
    validate_input(trial)
    return format_output(execute_logic(trial))


def main() -> None:
    """確認用。"""
    print(suggest_params(optuna.trial.FixedTrial({"kernel_size": 3})))
