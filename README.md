# sam2-mask-annotaion

A browser-based COCO instance mask correction tool that uses `datasets/annotations/wholebody48_person_body_coco.json` and the images under `datasets/images/`.

## Features

- FastAPI backend with a React/Vite frontend.
- Decodes and encodes COCO Compact RLE masks with `pycocotools`.
- Displays mask overlays with a different color per instance.
- Supports SAM2.1-assisted correction, SAM2 model switching from the UI, 1 px brush editing, undo, Ctrl + wheel zoom, image navigation, and index-based image jumping.
- In normal mode, left click or drag paints the mask, and right click or drag erases it. When the SAM2 support toggle is on, left click runs SAM2 assistance.
- Supports deleting the selected instance. A confirmation dialog is shown before deletion.
- Includes a reset button that restores all mask edits to the input COCO JSON state. A confirmation dialog is shown before reset.
- The folder button to the left of the save button opens another COCO annotation JSON file. Because this discards the current workspace state, a confirmation dialog is shown before loading.
- Corrected data can be saved to `datasets/annotations/wholebody48_person_body_coco.corrected.json`, and the same JSON can be downloaded from the browser.
- When another annotation JSON is opened, the save target becomes `datasets/annotations/<opened-file-name>.corrected.json`.
- During mask editing, only `area` is recalculated. `bbox` is preserved for both existing and new instances.

## Setup

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv sync --active --dev --extra sam2
python scripts/setup_sam2.py

cd frontend
pnpm install
```

The default model is `sam2.1_hiera_tiny.pt`. The UI can switch between `Tiny`, `Small`, `Base+`, and `Large`. Missing checkpoints are downloaded automatically when the SAM2 support toggle is turned on. If SAM2 support is already on and you switch models, the selected checkpoint is prepared as part of that switch.

To download checkpoints ahead of time, run:

```bash
python scripts/setup_sam2.py --model tiny
python scripts/setup_sam2.py --model all
```

To change the checkpoint directory or default model, set these environment variables:

```bash
export SAM2_CHECKPOINT_DIR=/path/to/checkpoints
export SAM2_DEFAULT_MODEL_ID=small
```

## Run

After setup, start both the backend and frontend with:

```bash
./scripts/dev.sh
```

Open `http://127.0.0.1:8999` in your browser.

If an existing dev server from this repository is already using the target ports, the script stops it and starts a fresh one.

To override the ports, set environment variables:

```bash
BACKEND_PORT=8010 FRONTEND_PORT=9000 ./scripts/dev.sh
```

To run the services separately:

```bash
source .venv/bin/activate
uvicorn backend.app.main:app --host 127.0.0.1 --port 8989 --reload

cd frontend
pnpm run dev
```

## Test

```bash
source .venv/bin/activate
pytest
cd frontend
pnpm run build
```
