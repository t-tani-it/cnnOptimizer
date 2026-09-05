# Why: 再現性確保 / What: seed固定 / Assumption: numpy/sklearn/torch混在可 / I-O: seed(int)->None / Caution: GPU完全再現は不可 / Future: なし / Change Log: 2026-09-05 new
"""seed固定ユーティリティ。"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 2) -> None:
    """seedを固定する。

    Args:
        seed: 乱数seed。

    Returns:
        None。

    Side Effects:
        os/numpy/randomへ設定。torchがあれば設定。

    Examples:
        >>> set_seed(2)
    """
    # 解説: 旧random_seed=2を継承
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
