# Why: 旧Sequential構造を再設計 / What: Conv-BN-Act-Pool-Drop構築 / Assumption: keras3 / I-O: params dict->Model / Caution: Test使用禁止 / Future: ブロック拡張 / Change Log: 2026-09-05 new
"""Kerasモデル構築。基本構造は旧思想を継承。"""

from __future__ import annotations

import keras
from keras import layers

from .mish import Mish


def validate_input(params: dict) -> None:
    """params検査。

    Args:
        params: ハイパーパラメータ。

    Raises:
        ValueError: 不正時。
    """
    if params["dropout"] < 0 or params["dropout"] >= 1:
        raise ValueError("invalid dropout")


def _act(name: str):
    # 解説: 文字列分岐を層に変換。tanhは組込み、mishのみ自作層
    if name == "mish":
        return Mish()
    return layers.Activation(name)


def execute_logic(params: dict) -> keras.Model:
    """モデル組立。

    Args:
        params: filters・kernel・dense・dropout・activation等。

    Returns:
        コンパイル前Model。
    """
    ks = (params["kernel_size"], params["kernel_size"])
    inp = layers.Input(shape=(28, 28, 1))
    x = inp
    # 解説: 旧fit_generator前処理を前処理層化しfit一本化
    if params.get("rotation", 0) or params.get("shift", 0) or params.get("zoom", 0):
        x = layers.RandomRotation(params["rotation"] / 360.0)(x)
        s = params["shift"]
        x = layers.RandomTranslation(s, s)(x)
        z = params["zoom"]
        x = layers.RandomZoom((-z, z))(x)
    filters = [params["conv1_filters"]] + [params["conv2_filters"]] * (
        params["n_conv_blocks"] - 1
    )
    for i, f in enumerate(filters):
        # 解説: Conv->BN->Act->Pool->Dropの旧基本構造を維持
        x = layers.Conv2D(int(f), ks, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = _act(params["activation"])(x)
        if i < len(filters) - 1:  # 解説: 最終Conv後は解像度保持(旧再現)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(params["dropout"])(x)
    x = layers.Flatten()(x)
    x = layers.Dense(int(params["dense_units"]))(x)
    x = _act(params["activation"])(x)
    out = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inp, out)


def format_output(model: keras.Model, params: dict) -> keras.Model:
    """optimizer・compile付与。

    Args:
        model: 未compile。
        params: optimizer・lr。

    Returns:
        compile済Model。
    """
    # 解説: 旧adam/amsgrad→Adam/AdamWへ現代化
    opt = (
        keras.optimizers.Adam(params["lr"])
        if params["optimizer"] == "adam"
        else keras.optimizers.AdamW(params["lr"])
    )
    model.compile(
        optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_model(params: dict) -> keras.Model:
    """公開口。

    Args:
        params: ハイパーパラメータ。

    Returns:
        学習可能Model。
    """
    validate_input(params)
    return format_output(execute_logic(params), params)


def main() -> None:
    """確認用。"""
    m = build_model(
        {
            "conv1_filters": 64,
            "conv2_filters": 128,
            "n_conv_blocks": 2,
            "kernel_size": 3,
            "dense_units": 256,
            "dropout": 0.25,
            "activation": "relu",
            "optimizer": "adam",
            "lr": 1e-3,
            "rotation": 10.0,
            "shift": 0.1,
            "zoom": 0.1,
        }
    )
    m.summary()
