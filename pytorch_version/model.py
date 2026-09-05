# Why: Kerasと同一構造をtorch化 / What: Conv-BN-Act-Pool-Drop / Assumption: torch2系 / I-O: params->nn.Module / Caution: Mishはnn.Mish使用 / Future: なし / Change Log: 2026-09-05 new
"""PyTorchモデル。"""

from __future__ import annotations

from torch import nn


def validate_input(params: dict) -> None:
    """検査。"""
    if params["dropout"] < 0 or params["dropout"] >= 1:
        raise ValueError("invalid dropout")


def _act(name: str) -> nn.Module:
    # 解説: 両FWで同一3種を保証。Mishは現代標準nn.Mish
    if name == "mish":
        return nn.Mish()
    if name == "tanh":
        return nn.Tanh()
    return nn.ReLU()


class CNN(nn.Module):
    """旧思想継承CNN。"""

    def __init__(self, params: dict):
        """初期化。

        Args:
            params: ハイパーパラメータ。
        """
        super().__init__()
        k = params["kernel_size"]
        pad = k // 2
        filters = [params["conv1_filters"]] + [params["conv2_filters"]] * (
            params["n_conv_blocks"] - 1
        )
        blocks: list = []
        in_c = 1
        for i, f in enumerate(filters):
            # 解説: Conv->BN->Act->Pool->Drop基本形
            blocks.append(nn.Conv2d(in_c, int(f), k, padding=pad))
            blocks.append(nn.BatchNorm2d(int(f)))
            blocks.append(_act(params["activation"]))
            if i < len(filters) - 1:
                blocks.append(nn.MaxPool2d(2))
                blocks.append(nn.Dropout(params["dropout"]))
            in_c = int(f)
        self.features = nn.Sequential(*blocks)
        # 解説: 28→n_pool回半減後の特徴量を自動算出
        with __import__("torch").no_grad():
            import torch

            d = torch.zeros(1, 1, 28, 28)
            n = self.features(d).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n, int(params["dense_units"])),
            _act(params["activation"]),
            nn.Linear(int(params["dense_units"]), 10),
        )

    def forward(self, x):
        """順伝播。"""
        return self.classifier(self.features(x))


def build_model(params: dict) -> CNN:
    """公開口。"""
    validate_input(params)
    return CNN(params)


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
        }
    )
    print(m)
