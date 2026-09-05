# Why: torch側Optuna探索 / What: suggest・学習・prune・保存 / Assumption: torchvision v2系 / I-O: trial->val_acc / Caution: AugmentはTrainのみ / Future: なし / Change Log: 2026-09-05 new
"""PyTorch用objective。"""

from __future__ import annotations

import time

import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
from .train import device, execute_logic, format_output

_DATA: tuple | None = None


def suggest_params(trial: optuna.Trial) -> dict:
    """Kerasと同一範囲提示。

    Args:
        trial: trial。

    Returns:
        params。
    """
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


def _loaders(params: dict):
    global _DATA
    if _DATA is None:
        _DATA = load_mnist_numpy()
    xtr, xva, _, ytr, yva, _ = _DATA
    # 解説: torchvision v2 functionalで回転・平行移動・ZoomをTrainのみ適用
    import torchvision.transforms.v2.functional as F

    def apply_aug(imgs: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        outs = []
        for im in t:
            if params["rotation"]:
                im = F.rotate(
                    im,
                    float(np.random.uniform(-params["rotation"], params["rotation"])),
                )
            if params["shift"]:
                s = params["shift"]
                im = F.affine(
                    im,
                    angle=0.0,
                    translate=(
                        float(np.random.uniform(-s, s)),
                        float(np.random.uniform(-s, s)),
                    ),
                    scale=1.0,
                    shear=0.0,
                )
            if params["zoom"]:
                z = 1.0 + float(np.random.uniform(-params["zoom"], params["zoom"]))
                im = F.affine(im, angle=0.0, translate=(0.0, 0.0), scale=z, shear=0.0)
            outs.append(im)
        return torch.stack(outs).float()

    xtr_t = apply_aug(xtr)
    xva_t = torch.from_numpy(xva).permute(0, 3, 1, 2).float()
    ytr_t = torch.from_numpy(ytr).long()
    yva_t = torch.from_numpy(yva).long()
    tr = DataLoader(
        TensorDataset(xtr_t, ytr_t), batch_size=params["batch_size"], shuffle=True
    )
    va = DataLoader(TensorDataset(xva_t, yva_t), batch_size=512)
    return tr, va


def objective(trial: optuna.Trial) -> float:
    """1 Trial実行。

    Args:
        trial: trial。

    Returns:
        val accuracy。
    """
    params = suggest_params(trial)
    tr, va = _loaders(params)
    dev = device()
    model = build_model(params).to(dev)
    crit = nn.CrossEntropyLoss()
    opt = (
        torch.optim.Adam(model.parameters(), lr=params["lr"])
        if params["optimizer"] == "adam"
        else torch.optim.AdamW(model.parameters(), lr=params["lr"])
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR
    )
    best_acc, best_loss, bad, t0 = 0.0, 1e9, 0, time.time()
    for epoch in range(MAX_EPOCHS):
        execute_logic(model, tr, opt, crit)
        vloss, vacc = format_output(model, va, crit)
        sched.step(vloss)
        # 解説: epoch毎報告し不良TrialをPruning
        trial.report(vacc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if vloss < best_loss:
            best_loss, best_acc, bad = vloss, vacc, 0
            torch.save(
                model.state_dict(), f"models/torch_trial{trial.number}_{vacc:.4f}.pt"
            )
        else:
            bad += 1
            if bad >= EARLY_PATIENCE:
                break
    save_trial(
        RESULTS_DB,
        {
            "framework": "torch",
            "params": params,
            "val_acc": best_acc,
            "val_loss": best_loss,
            "elapsed": time.time() - t0,
        },
    )
    return best_acc
