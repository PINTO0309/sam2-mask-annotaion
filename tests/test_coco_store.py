from __future__ import annotations

import json

import numpy as np

from backend.app.rle import compact_rle_to_mask


def test_existing_annotation_bbox_is_preserved(sample_store):
    replacement = np.zeros((5, 6), dtype=bool)
    replacement[0:2, 0:2] = True

    sample_store.update_mask(10, replacement)
    sample_store.save()
    saved = json.loads(sample_store.output_path.read_text(encoding="utf-8"))
    annotation = saved["annotations"][0]

    assert annotation["bbox"] == [2.25, 1.25, 2.5, 2.5]
    assert annotation["area"] == 4.0
    assert isinstance(annotation["segmentation"]["counts"], str)
    assert compact_rle_to_mask(annotation["segmentation"]).sum() == 4


def test_new_annotation_on_extra_image_adds_image_and_computed_bbox(sample_store):
    mask = np.zeros((5, 6), dtype=bool)
    mask[2:5, 1:4] = True

    annotation = sample_store.create_instance(2, mask)
    sample_store.save()
    saved = json.loads(sample_store.output_path.read_text(encoding="utf-8"))

    assert annotation["id"] == 11
    assert annotation["image_id"] == 2
    assert saved["images"][1] == {"id": 2, "file_name": "extra.png", "width": 6, "height": 5}
    assert saved["annotations"][1]["bbox"] == [1.0, 2.0, 3.0, 3.0]
    assert saved["annotations"][1]["area"] == 9.0
    assert isinstance(saved["annotations"][1]["segmentation"]["counts"], str)
