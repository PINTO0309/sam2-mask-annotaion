from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict


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

SAM2_CHECKPOINT_DIR = Path(os.environ.get("SAM2_CHECKPOINT_DIR", ROOT_DIR / "checkpoints"))
SAM2_DEFAULT_MODEL_ID = os.environ.get("SAM2_DEFAULT_MODEL_ID", "tiny")


class SAM2ModelSpec(TypedDict):
    id: str
    label: str
    checkpoint_url: str
    checkpoint_path: Path
    model_cfg: str


SAM2_MODELS: dict[str, SAM2ModelSpec] = {
    "tiny": {
        "id": "tiny",
        "label": "Tiny",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "checkpoint_path": Path(
            os.environ.get("SAM2_CHECKPOINT_PATH", SAM2_CHECKPOINT_DIR / "sam2.1_hiera_tiny.pt")
        ),
        "model_cfg": os.environ.get("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    },
    "small": {
        "id": "small",
        "label": "Small",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "checkpoint_path": SAM2_CHECKPOINT_DIR / "sam2.1_hiera_small.pt",
        "model_cfg": "configs/sam2.1/sam2.1_hiera_s.yaml",
    },
    "base_plus": {
        "id": "base_plus",
        "label": "Base+",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "checkpoint_path": SAM2_CHECKPOINT_DIR / "sam2.1_hiera_base_plus.pt",
        "model_cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    },
    "large": {
        "id": "large",
        "label": "Large",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "checkpoint_path": SAM2_CHECKPOINT_DIR / "sam2.1_hiera_large.pt",
        "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
    },
}
