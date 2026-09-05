# Why: Optuna結果・重要度の可視化 / What: png出力 / Assumption: optuna/matplotlib導入 / I-O: study->png / Caution: 重い描画はtry回避 / Future: html report / Change Log: 2026-09-05 new
"""可視化ヘルパー。"""

from __future__ import annotations

import optuna


def main(study: optuna.Study, outdir: str = "results") -> None:
    """Study可視化を保存する。

    Args:
        study: 完了済Study。
        outdir: 出力先。

    Side Effects:
        png保存。失敗しても例外化しない。

    Examples:
        >>> main(study)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.get_figure().savefig(f"{outdir}/history_{study.study_name}.png")
        plt.close("all")
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        fig.get_figure().savefig(f"{outdir}/importance_{study.study_name}.png")
        plt.close("all")
    except Exception as e:  # noqa: BLE001 解説: Trial不足でも全体を止めないため
        print(f"visualize skipped: {e}")
