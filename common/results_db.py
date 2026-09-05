# Why: TinyDB/TXT廃止しSQLite一本化 / What: Trial結果保存・読出 / Assumption: sqlite3標準のみ / I-O: dict入出力 / Caution: 同時書込は直列化 / Future: mlflow移行可 / Change Log: 2026-09-05 new
"""SQLite結果DB。Optuna Storageとは別に集計用テーブルを持つ。"""

from __future__ import annotations

import json
import sqlite3
import time


def validate_input(record: dict) -> None:
    """必須キー検査。

    Args:
        record: 保存レコード。

    Raises:
        ValueError: 欠落時。
    """
    for k in ("framework", "val_acc", "val_loss", "elapsed"):
        if k not in record:
            raise ValueError(f"missing {k}")


def execute_logic(db_path: str, record: dict) -> int:
    """保存実行。

    Args:
        db_path: sqliteパス。
        record: trial情報(params含む)。

    Returns:
        rowid。
    """
    con = sqlite3.connect(db_path)
    con.execute(
        """CREATE TABLE IF NOT EXISTS trials(
        id INTEGER PRIMARY KEY AUTOINCREMENT, framework TEXT,
        params TEXT, val_acc REAL, val_loss REAL, elapsed REAL, ts REAL)"""
    )
    cur = con.execute(
        "INSERT INTO trials(framework,params,val_acc,val_loss,elapsed,ts) VALUES(?,?,?,?,?,?)",
        (
            record["framework"],
            json.dumps(record.get("params", {})),
            float(record["val_acc"]),
            float(record["val_loss"]),
            float(record["elapsed"]),
            time.time(),
        ),
    )
    con.commit()
    rid = cur.lastrowid or 0
    con.close()
    return rid


def format_output(row: tuple) -> dict:
    """行整形。

    Args:
        row: DB行。

    Returns:
        dictレコード。
    """
    return {
        "id": row[0],
        "framework": row[1],
        "params": json.loads(row[2]),
        "val_acc": row[3],
        "val_loss": row[4],
        "elapsed": row[5],
    }


def save_trial(db_path: str, record: dict) -> int:
    """Trial保存口。

    Args:
        db_path: sqliteパス。
        record: レコード。

    Returns:
        rowid。
    """
    validate_input(record)
    return execute_logic(db_path, record)


def best_trial(db_path: str, framework: str) -> dict | None:
    """最良Trial取得。

    Args:
        db_path: sqliteパス。
        framework: keras/torch。

    Returns:
        最良レコード、無ければNone。
    """
    con = sqlite3.connect(db_path)
    con.execute(
        """CREATE TABLE IF NOT EXISTS trials(
        id INTEGER PRIMARY KEY AUTOINCREMENT, framework TEXT,
        params TEXT, val_acc REAL, val_loss REAL, elapsed REAL, ts REAL)"""
    )
    cur = con.execute(
        "SELECT * FROM trials WHERE framework=? ORDER BY val_acc DESC LIMIT 1",
        (framework,),
    )
    row = cur.fetchone()
    con.close()
    return format_output(row) if row else None


def main() -> None:
    """確認用。"""
    print(best_trial("results/results.db", "keras"))
