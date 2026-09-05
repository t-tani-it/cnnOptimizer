# Why: Keras実験入口 / What: Study作成・最適化・可視化 / Assumption: optuna4系 / I-O: CLI->best val_acc / Caution: 継続は同study名+storage / Future: なし / Change Log: 2026-09-05 new
"""Keras Optuna実行。"""

from __future__ import annotations

import argparse

import optuna

from common.config import N_TRIALS_DEFAULT, OPTUNA_DB
from common.seed import set_seed
from common.visualize import main as viz

from .objective import objective


def validate_input(args: argparse.Namespace) -> None:
    """引数検査。"""
    if args.n_trials <= 0:
        raise ValueError("n_trials must be >0")


def execute_logic(args: argparse.Namespace) -> optuna.Study:
    """最適化実行。

    Args:
        args: n_trials・study名等。

    Returns:
        完了Study。
    """
    set_seed()
    # 解説: 総当たりでなくTPE、MedianPrunerで早期打切り
    study = optuna.create_study(
        study_name=args.study,
        storage=f"sqlite:///{OPTUNA_DB}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=args.n_trials)
    return study


def format_output(study: optuna.Study) -> None:
    """結果表示・保存。

    Args:
        study: 完了Study。

    Side Effects:
        printとpng保存。
    """
    print(f"best={study.best_value:.4f} params={study.best_params}")
    viz(study)


def main(args: argparse.Namespace | None = None) -> None:
    """CLI入口。

    Side Effects:
        最適化実行。
    """
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    p.add_argument("--study", type=str, default="mnist-keras")
    a = p.parse_args() if args is None else args
    validate_input(a)
    format_output(execute_logic(a))


if __name__ == "__main__":
    main()
