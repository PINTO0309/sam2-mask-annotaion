from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.app.coco_store import COCOStore
from backend.app.rle import mask_to_compact_rle


@pytest.fixture()
def sample_dataset(tmp_path: Path):
    images_dir = tmp_path / "datasets" / "images"
    annotations_dir = tmp_path / "datasets" / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    Image.new("RGB", (6, 5), (120, 120, 120)).save(images_dir / "000001.jpg")
    Image.new("RGB", (6, 5), (80, 80, 80)).save(images_dir / "extra.png")

    mask = np.zeros((5, 6), dtype=bool)
    mask[1:4, 2:5] = True
    coco = {
        "info": {},
        "licenses": [],
        "images": [{"id": 1, "file_name": "000001.jpg", "width": 6, "height": 5}],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "segmentation": mask_to_compact_rle(mask),
                "area": 9.0,
                "bbox": [2.25, 1.25, 2.5, 2.5],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "person_body", "supercategory": "person"}],
    }
    coco_path = annotations_dir / "wholebody48_person_body_coco.json"
    output_path = annotations_dir / "wholebody48_person_body_coco.corrected.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")
    return coco_path, images_dir, output_path


@pytest.fixture()
def sample_store(sample_dataset):
    coco_path, images_dir, output_path = sample_dataset
    return COCOStore(coco_path, images_dir, output_path)
