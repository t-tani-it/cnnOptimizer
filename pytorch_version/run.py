# Why: torch実験入口 / What: Study最適化 / Assumption: optuna4系 / I-O: CLI->best / Caution: 同名継続可 / Future: なし / Change Log: 2026-09-05 new
"""PyTorch実行。"""

from __future__ import annotations

import argparse

import optuna

from common.config import N_TRIALS_DEFAULT, OPTUNA_DB
from common.seed import set_seed
from common.visualize import main as viz

from .objective import objective


def main() -> None:
    """CLI入口。

    Side Effects:
        最適化と可視化。
    """
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    p.add_argument("--study", type=str, default="mnist-torch")
    a = p.parse_args()
    if a.n_trials <= 0:
        raise ValueError("n_trials must be >0")
    set_seed()
    study = optuna.create_study(
        study_name=a.study,
        storage=f"sqlite:///{OPTUNA_DB}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=a.n_trials)
    print(f"best={study.best_value:.4f} params={study.best_params}")
    viz(study)


if __name__ == "__main__":
    main()
