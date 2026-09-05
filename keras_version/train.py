# Why: 最良条件の再学習・Test評価 / What: best再訓練 / Assumption: results.db存在 / I-O: best params->test acc / Caution: Testはここでのみ使用 / Future: なし / Change Log: 2026-09-05 new
"""Keras最良モデル再学習。"""

from __future__ import annotations

import keras

from common.config import EARLY_PATIENCE, MAX_EPOCHS, RESULTS_DB
from common.data import load_mnist_numpy
from common.results_db import best_trial

from .model import build_model


def main() -> float:
    """最良paramsで再学習しTest評価する。

    Returns:
        test accuracy。

    Side Effects:
        models/keras_best.keras保存。
    """
    best = best_trial(RESULTS_DB, "keras")
    if best is None:
        raise RuntimeError("no keras trials")
    params = best["params"]
    xtr, xva, xte, ytr, yva, yte = load_mnist_numpy()
    import numpy as np

    xa = np.concatenate([xtr, xva])
    ya = np.concatenate([ytr, yva])
    model = build_model(params)
    model.fit(
        xa,
        ya,
        epochs=MAX_EPOCHS,
        batch_size=params["batch_size"],
        verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="loss", patience=EARLY_PATIENCE, restore_best_weights=True
            )
        ],
    )
    loss, acc = model.evaluate(xte, yte, verbose=0)
    model.save("models/keras_best.keras")
    print(f"test acc={acc:.4f} loss={loss:.4f}")
    return float(acc)
