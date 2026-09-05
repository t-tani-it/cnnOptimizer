# cnnOptimizer AGENTS.md

## 概要
MNIST CNN精度向上実験をKeras 3 + OptunaとPyTorch + Optunaで再構築する。

## 開発環境
- OS: Windows、前提Python 3.10-3.11、CUDA利用可なら自動使用
- 対象env例: dynamicAnalysis (py3.11)

## セキュリティ
- ローカルパス・APIキー・個人ID・トークンをコードに含めない
- TinyDB/TXT旧保存方式は廃止、SQLiteのみ

## GitHub運用
- HTTPS + gh CLI、public、main追跡
- push前は禁止情報チェック、問題なければ「公開禁止情報は含まれていません」と表示
- 分割コミット、docs/はPush除外(.gitignore)
- フェーズ完了時はdocs/handover.md更新(ローカルのみ)

## 技術スタック
- keras>=3.10(tensorflow>=2.16 backend)、torch>=2.3、optuna>=4.0、scikit-learn、matplotlib
- DB: results/optuna.db(Study)、results/results.db(集計)

## 構成
- common/ → data・seed・results_db・visualize・config
- keras_version/、pytorch_version/ → model/train/objective/run分離
- sample/ → 旧コード保存、改変禁止

## コーディング規約
- ファイル先頭にWhy/What/Assumption/I-O/Caution/Future/ChangeLog
- 設定値はconfig.pyに集約
- main/validate_input/execute_logic/format_output構成、単一責務、型ヒント必須
- docstring(Sphinx形式、何をするか・引数・戻り値・副作用・例)、コメントはなぜを書く
- fit_generator等非推奨API禁止、公式ドキュメント準拠

## 実行・テスト
- Keras: python -m keras_version.run --n-trials 50
- PyTorch: python -m pytorch_version.run --n-trials 50
- テスト: pytest、Lint: ruff check .
