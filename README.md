# sam2-mask-annotaion

`datasets/annotations/wholebody48_person_body_coco.json` と `datasets/images/` を入力として使う、ブラウザベースの COCO インスタンスマスク補正ツールです。

## 機能

- FastAPI バックエンドと React/Vite フロントエンドで構成されています。
- `pycocotools` を使って COCO Compact RLE をデコード/エンコードします。
- インスタンスごとに異なる色でマスクをオーバーレイ表示します。
- SAM2.1 のポイントプロンプトによる補正、UI 上での SAM2 モデル切り替え、1 px まで指定できる消しゴム編集、UNDO、マウスホイールでのズーム、画像移動、インデックス指定ジャンプに対応しています。
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

デフォルトでは `sam2.1_hiera_tiny.pt` を使います。UI から `Tiny`、`Small`、`Base+`、`Large` を選択でき、未取得の checkpoint は選択後の初回推論時に自動ダウンロードされます。事前に取得する場合は次のように実行してください。

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

ターミナル 1:

```bash
source .venv/bin/activate
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

ターミナル 2:

```bash
cd frontend
pnpm run dev
```

ブラウザで `http://127.0.0.1:5173` を開いてください。

## テスト

```bash
source .venv/bin/activate
pytest
cd frontend
pnpm run build
```
