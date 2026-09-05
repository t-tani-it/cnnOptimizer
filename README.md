# mnist-cnn-optuna

4年前MNIST CNN実験の現代版再構築。Keras 3 + OptunaとPyTorch + Optunaで同一条件比較する。

## 概要
- MNIST 10分類、28x28x1、0-1正規化、Train/Val/Test分離
- 基本構造 Conv→BN→Act→Pool→Dropout継承
- TPE Sampler + MedianPruner、SQLite保存、GPU自動

## 環境構築
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 実行方法
```bash
python -m keras_version.run --n-trials 50
python -m pytorch_version.run --n-trials 50
python -m keras_version.train
```

## 最良Trial確認
```bash
python -c "from common.results_db import best_trial; print(best_trial('results/results.db','keras'))"
python -c "from common.results_db import best_trial; print(best_trial('results/results.db','torch'))"
```
Optuna UI代替: `results/history_*.png` `results/importance_*.png`参照。

## Keras/PyTorch比較
- 同一分割(seed2,val10%)・同一探索範囲・同一評価(val_acc最大化)
- `results/results.db`のtrials表でframework別に集計比較

## 対応関係・変更点
詳細は `docs/old_vs_modern.md`(ローカルのみ、Push除外)を参照。
要旨: 総当たり→TPE、fit_generator→fit/Loader、TXT/TinyDB→SQLite、Testリーク修正、Mish現代化。

## 旧コード(sample/)
4年前の実験コード保存場所。参照専用・改変禁止(公開のための可搬化1行を除く)。
詳細は `docs/old_vs_modern.md`(ローカルのみ)を参照。
