# sam2-mask-annotaion

`datasets/annotations/wholebody48_person_body_coco.json` と `datasets/images/` を入力として使う、ブラウザベースの COCO インスタンスマスク補正ツールです。

## 機能

- FastAPI バックエンドと React/Vite フロントエンドで構成されています。
- `pycocotools` を使って COCO Compact RLE をデコード/エンコードします。
- インスタンスごとに異なる色でマスクをオーバーレイ表示します。
- SAM2.1 による補助補正、UI 上での SAM2 モデル切り替え、1 px まで指定できるブラシ編集、UNDO、マウスホイールでのズーム、画像移動、インデックス指定ジャンプに対応しています。通常時は左クリック/ドラッグでマスクを塗り、右クリック/ドラッグでマスクを除去します。SAM2 サポートトグルが ON のときは、左クリックで SAM2 補助を実行します。
- 全マスク編集を入力 COCO JSON の状態へ戻す初期化ボタンを備えています。実行前には確認ダイアログを表示します。
- 補正後のデータは `datasets/annotations/wholebody48_person_body_coco.corrected.json` に保存でき、同じ JSON をブラウザからダウンロードできます。
- 既存アノテーションの `bbox` は保持し、新規インスタンスのみ編集後のマスクから `bbox` と `area` を計算します。

## セットアップ

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv sync --active --dev --extra sam2
python scripts/setup_sam2.py

cd frontend
pnpm install
```

デフォルトでは `sam2.1_hiera_tiny.pt` を使います。UI から `Tiny`、`Small`、`Base+`、`Large` を選択でき、未取得の checkpoint は SAM2 サポートトグルを ON にした時点で自動ダウンロードされます。SAM2 サポートが ON の状態でモデルを切り替えた場合も、そのモデルの checkpoint を準備します。事前に取得する場合は次のように実行してください。

```bash
python scripts/setup_sam2.py --model tiny
python scripts/setup_sam2.py --model all
```

checkpoint ディレクトリやデフォルトモデルを変更する場合は、次の環境変数を設定してください。

```bash
export SAM2_CHECKPOINT_DIR=/path/to/checkpoints
export SAM2_DEFAULT_MODEL_ID=small
```

## 起動

セットアップ後は、次のワンライナーでバックエンドとフロントエンドを同時に起動できます。

```bash
./scripts/dev.sh
```

ブラウザで `http://127.0.0.1:8999` を開いてください。
同じリポジトリから起動した既存の dev server が対象ポートを使用している場合は、自動で停止してから起動し直します。

ポートを変更する場合は環境変数を指定します。

```bash
BACKEND_PORT=8010 FRONTEND_PORT=9000 ./scripts/dev.sh
```

個別に起動する場合は次のコマンドを使います。

```bash
source .venv/bin/activate
uvicorn backend.app.main:app --host 127.0.0.1 --port 8989 --reload

cd frontend
pnpm run dev
```

## テスト

```bash
source .venv/bin/activate
pytest
cd frontend
pnpm run build
```
