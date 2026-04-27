# sam2-mask-annotaion

`datasets/annotations/wholebody48_person_body_coco.json` と `datasets/images/` を入力として使う、ブラウザベースの COCO インスタンスマスク補正ツールです。

## 機能

- FastAPI バックエンドと React/Vite フロントエンドで構成されています。
- `pycocotools` を使って COCO Compact RLE をデコード/エンコードします。
- インスタンスごとに異なる色でマスクをオーバーレイ表示します。
- SAM2.1 のポイントプロンプトによる補正、1 px まで指定できる消しゴム編集、UNDO、マウスホイールでのズーム、画像移動、インデックス指定ジャンプに対応しています。
- 補正後のデータは `datasets/annotations/wholebody48_person_body_coco.corrected.json` に保存でき、同じ JSON をブラウザからダウンロードできます。
- 既存アノテーションの `bbox` は保持し、新規インスタンスのみ編集後のマスクから `bbox` と `area` を計算します。

## セットアップ

```bash
uv sync --dev --extra sam2
uv run python scripts/setup_sam2.py

cd frontend
npm install
```

デフォルトの SAM2 checkpoint は `sam2.1_hiera_tiny.pt` です。パスを変更する場合は、次の環境変数を設定してください。

```bash
export SAM2_CHECKPOINT_PATH=/path/to/sam2.1_hiera_tiny.pt
export SAM2_MODEL_CFG=configs/sam2.1/sam2.1_hiera_t.yaml
```

## 起動

ターミナル 1:

```bash
uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

ターミナル 2:

```bash
cd frontend
npm run dev
```

ブラウザで `http://127.0.0.1:5173` を開いてください。

## テスト

```bash
uv run pytest
cd frontend
npm run build
```
