# Why: 旧自作Mishを現代API化 / What: Mish層 / Assumption: keras3+TF / I-O: tensor->tensor / Caution: get_custom_objects登録要 / Future: なし / Change Log: 2026-09-05 new
"""Mish活性化の現代実装。"""

from __future__ import annotations

import keras
from keras import layers


@keras.saving.register_keras_serializable()
def mish_fn(x):
    """Mish関数 x*tanh(softplus(x))。

    Args:
        x: テンソル。

    Returns:
        変換後テンソル。
    """
    import tensorflow as tf  # 解説: keras ops依存を避けTF直参照で安定化

    return x * tf.math.tanh(tf.math.softplus(x))


class Mish(layers.Layer):
    """Mish層。"""

    def call(self, x):
        """順伝播。

        Args:
            x: 入力。

        Returns:
            Mish適用結果。
        """
        return mish_fn(x)
