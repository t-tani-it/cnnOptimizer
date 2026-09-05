# Why: 軽量検証 / What: config・DB・seed試験 / Assumption: pytest / I-O: assert / Caution: 重い学習なし / Future: なし / Change Log: 2026-09-05 new
"""軽量テスト。"""

import os
import tempfile

from common.config import SEARCH_SPACE
from common.results_db import best_trial, save_trial
from common.seed import set_seed


def test_search_space_has_keys():
    """範囲キー存在。"""
    for k in ("dropout", "activation", "optimizer", "lr"):
        assert k in SEARCH_SPACE


def test_seed_and_db(tmp_path=None):
    """seedとDB保存。"""
    set_seed(2)
    fd, p = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    rid = save_trial(
        p,
        {
            "framework": "keras",
            "params": {"lr": 0.001},
            "val_acc": 0.9,
            "val_loss": 0.3,
            "elapsed": 1.0,
        },
    )
    assert rid >= 1
    b = best_trial(p, "keras")
    assert b is not None and b["val_acc"] == 0.9
    os.remove(p)
