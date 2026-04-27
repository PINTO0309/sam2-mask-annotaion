from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASETS_DIR = Path(os.environ.get("SAM2_TOOL_DATASETS_DIR", ROOT_DIR / "datasets"))
IMAGES_DIR = Path(os.environ.get("SAM2_TOOL_IMAGES_DIR", DATASETS_DIR / "images"))
ANNOTATIONS_DIR = Path(os.environ.get("SAM2_TOOL_ANNOTATIONS_DIR", DATASETS_DIR / "annotations"))
SOURCE_COCO_PATH = Path(
    os.environ.get(
        "SAM2_TOOL_COCO_PATH",
        ANNOTATIONS_DIR / "wholebody48_person_body_coco.json",
    )
)
OUTPUT_COCO_PATH = Path(
    os.environ.get(
        "SAM2_TOOL_OUTPUT_COCO_PATH",
        ANNOTATIONS_DIR / "wholebody48_person_body_coco.corrected.json",
    )
)

SAM2_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
SAM2_CHECKPOINT_PATH = Path(
    os.environ.get("SAM2_CHECKPOINT_PATH", ROOT_DIR / "checkpoints" / "sam2.1_hiera_tiny.pt")
)
SAM2_MODEL_CFG = os.environ.get("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_t.yaml")
