# Why: 両FWで同一分割・同一評価を保証 / What: MNIST取得・分割 / Assumption: keras/torchどちらか利用可 / I-O: arrays分割済 / Caution: Testは探索使用禁止 / Future: FashionMNIST対応 / Change Log: 2026-09-05 new
"""MNIST共通データ処理。"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from .config import RANDOM_SEED, VAL_SIZE


def validate_input(x_train: np.ndarray, y_train: np.ndarray) -> None:
    """入力チェック。

    Args:
        x_train: 画像配列。
        y_train: ラベル配列。

    Raises:
        ValueError: 空または不整合時。
    """
    if len(x_train) == 0 or len(x_train) != len(y_train):
        raise ValueError("invalid train data")


def execute_logic(
    x_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train/Val分割を実行。

    Args:
        x_train: 0-1正規化済画像(N,28,28,1)。
        y_train: ラベル(N,)。

    Returns:
        Xtr, Xva, ytr, yvaのtuple。
    """
    # 解説: 旧test_size=0.1, random_state=2を継承
    return train_test_split(
        x_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_train
    )


def format_output(arr: np.ndarray) -> np.ndarray:
    """float32化の整形。

    Args:
        arr: 配列。

    Returns:
        float32配列。
    """
    return arr.astype("float32")


def load_mnist_numpy() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """MNISTを読込みTrain/Val/Testに分離する。

    Returns:
        Xtr, Xva, Xte, ytr, yva, yte。正規化済。

    Side Effects:
        初回はダウンロード発生。
    """
    from keras.datasets import mnist  # 解説: 両FW共通の取得口として利用

    (xtr, ytr), (xte, yte) = mnist.load_data()
    xtr = format_output(xtr.reshape(-1, 28, 28, 1) / 255.0)
    xte = format_output(xte.reshape(-1, 28, 28, 1) / 255.0)
    validate_input(xtr, ytr)
    xtr, xva, ytr, yva = execute_logic(xtr, ytr)
    return xtr, xva, xte, ytr, yva, yte


def main() -> None:
    """動作確認用。

    Side Effects:
        形状をprint。
    """
    xtr, xva, xte, _, _, _ = load_mnist_numpy()
    print(xtr.shape, xva.shape, xte.shape)
