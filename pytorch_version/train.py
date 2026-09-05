# Why: epoch学習共通化 / What: 1epoch・評価・早期終了 / Assumption: CUDA自動 / I-O: loader->loss/acc / Caution: Testは評価専用 / Future: AMP対応 / Change Log: 2026-09-05 new
"""PyTorch学習ループ。"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def device() -> torch.device:
    """GPU自動選択。

    Returns:
        device。
    """
    # 解説: 利用可ならGPUの決まりを自動化
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_input(model: nn.Module, loader: DataLoader) -> None:
    """検査。"""
    if len(loader) == 0:
        raise ValueError("empty loader")


def execute_logic(
    model: nn.Module, loader: DataLoader, opt, crit
) -> tuple[float, float]:
    """1epoch学習。

    Args:
        model: モデル。opt: optimizer。crit: 損失。

    Returns:
        (loss, acc)。
    """
    model.train()
    dev = next(model.parameters()).device
    cor, loss_sum, n = 0, 0.0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * len(y)
        cor += (out.argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, cor / n


def format_output(model: nn.Module, loader: DataLoader, crit) -> tuple[float, float]:
    """検証評価。

    Args:
        model: モデル。loader: 検証。crit: 損失。

    Returns:
        (loss, acc)。
    """
    model.eval()
    dev = next(model.parameters()).device
    loss_sum, cor, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss_sum += crit(out, y).item() * len(y)
            cor += (out.argmax(1) == y).sum().item()
            n += len(y)
    return loss_sum / n, cor / n


def main() -> None:
    """確認用。"""
    print(device())
