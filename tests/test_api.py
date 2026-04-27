from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.rle import mask_to_compact_rle, mask_to_png_data_url


class FakeSAM2:
    def __init__(self):
        self.current_model_id = "tiny"
        self.last_model_id = None

    def list_models(self):
        return {
            "current_model_id": self.current_model_id,
            "models": [
                {"id": "tiny", "label": "Tiny", "checkpoint_path": "tiny.pt", "available": True},
                {"id": "small", "label": "Small", "checkpoint_path": "small.pt", "available": False},
            ],
        }

    def select_model(self, model_id):
        if model_id not in {"tiny", "small"}:
            raise ValueError(f"unknown SAM2 model: {model_id}")
        self.current_model_id = model_id
        return self.list_models()

    def prepare_model(self, model_id=None):
        if model_id:
            self.select_model(model_id)
        return {
            "current_model_id": self.current_model_id,
            "checkpoint_path": f"{self.current_model_id}.pt",
            "available": True,
            "models": self.list_models()["models"],
        }

    def predict(self, image_path, points, model_id=None):
        self.last_model_id = model_id
        mask = np.zeros((5, 6), dtype=bool)
        mask[1:3, 1:3] = True
        return mask


def test_image_list_detail_update_save_and_download(sample_store):
    app = create_app(store=sample_store, sam2_service=FakeSAM2())
    client = TestClient(app)

    images = client.get("/api/images").json()["images"]
    assert [image["file_name"] for image in images] == ["000001.jpg", "extra.png"]
    assert images[0]["annotation_count"] == 1
    assert images[1]["in_coco"] is False

    detail = client.get("/api/images/1").json()
    assert detail["annotations"][0]["id"] == 10
    assert detail["annotations"][0]["mask_png"].startswith("data:image/png;base64,")

    mask = np.zeros((5, 6), dtype=bool)
    mask[0, 0] = True
    response = client.put("/api/annotations/10/mask", json={"mask_png": mask_to_png_data_url(mask)})
    assert response.status_code == 200
    assert response.json()["bbox"] == [2.25, 1.25, 2.5, 2.5]

    assert client.post("/api/save").status_code == 200
    download = client.get("/api/download")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("application/json")


def test_sam2_predict_endpoint_uses_service(sample_store):
    sam2 = FakeSAM2()
    app = create_app(store=sample_store, sam2_service=sam2)
    client = TestClient(app)

    response = client.post(
        "/api/sam2/predict",
        json={"image_index": 1, "points": [{"x": 1, "y": 1, "label": 1}], "model_id": "small"},
    )

    assert response.status_code == 200
    assert response.json()["mask_png"].startswith("data:image/png;base64,")
    assert sam2.last_model_id == "small"


def test_delete_annotation_endpoint(sample_store):
    app = create_app(store=sample_store, sam2_service=FakeSAM2())
    client = TestClient(app)

    response = client.delete("/api/annotations/10")
    assert response.status_code == 200
    assert response.json() == {"deleted_id": 10, "image_id": 1}
    assert client.get("/api/images/1").json()["annotations"] == []

    missing = client.delete("/api/annotations/10")
    assert missing.status_code == 404


def test_sam2_model_list_and_select(sample_store):
    app = create_app(store=sample_store, sam2_service=FakeSAM2())
    client = TestClient(app)

    models = client.get("/api/sam2/models")
    assert models.status_code == 200
    assert models.json()["current_model_id"] == "tiny"

    selected = client.post("/api/sam2/models/select", json={"model_id": "small"})
    assert selected.status_code == 200
    assert selected.json()["current_model_id"] == "small"

    invalid = client.post("/api/sam2/models/select", json={"model_id": "missing"})
    assert invalid.status_code == 400


def test_sam2_model_prepare(sample_store):
    app = create_app(store=sample_store, sam2_service=FakeSAM2())
    client = TestClient(app)

    prepared = client.post("/api/sam2/models/prepare", json={"model_id": "small"})
    assert prepared.status_code == 200
    assert prepared.json()["current_model_id"] == "small"
    assert prepared.json()["available"] is True

    invalid = client.post("/api/sam2/models/prepare", json={"model_id": "missing"})
    assert invalid.status_code == 400


def test_reset_edits_reloads_source_coco_and_removes_new_instances(sample_store):
    app = create_app(store=sample_store, sam2_service=FakeSAM2())
    client = TestClient(app)

    changed = np.zeros((5, 6), dtype=bool)
    changed[0, 0] = True
    assert client.put("/api/annotations/10/mask", json={"mask_png": mask_to_png_data_url(changed)}).status_code == 200
    assert client.post("/api/images/2/instances", json={}).status_code == 200

    reset = client.post("/api/reset-edits")
    assert reset.status_code == 200

    first_detail = client.get("/api/images/1").json()
    extra_detail = client.get("/api/images/2").json()
    assert first_detail["annotations"][0]["area"] == 9.0
    assert extra_detail["annotations"] == []
    assert extra_detail["in_coco"] is False


def test_open_annotation_file_changes_source_save_target_and_reset(sample_store):
    app = create_app(store=sample_store, sam2_service=FakeSAM2())
    client = TestClient(app)

    opened_mask = np.zeros((5, 6), dtype=bool)
    opened_mask[0, 0:2] = True
    opened_coco = {
        "info": {},
        "licenses": [],
        "images": [{"id": 7, "file_name": "extra.png", "width": 6, "height": 5}],
        "annotations": [
            {
                "id": 20,
                "image_id": 7,
                "category_id": 1,
                "segmentation": mask_to_compact_rle(opened_mask),
                "area": 2.0,
                "bbox": [0.0, 0.0, 2.0, 1.0],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "person_body", "supercategory": "person"}],
    }

    opened = client.post("/api/annotations/open", json={"file_name": "alternate.json", "data": opened_coco})
    assert opened.status_code == 200
    assert opened.json()["source_name"] == "alternate.json"
    assert opened.json()["output_path"].endswith("alternate.corrected.json")

    images = client.get("/api/images").json()["images"]
    assert [image["file_name"] for image in images] == ["extra.png", "000001.jpg"]
    first_detail = client.get("/api/images/1").json()
    assert first_detail["annotations"][0]["id"] == 20
    assert first_detail["annotations"][0]["area"] == 2.0

    changed = np.zeros((5, 6), dtype=bool)
    changed[1:4, 1:4] = True
    assert client.put("/api/annotations/20/mask", json={"mask_png": mask_to_png_data_url(changed)}).status_code == 200
    assert client.get("/api/images/1").json()["annotations"][0]["area"] == 9.0

    assert client.post("/api/reset-edits").status_code == 200
    assert client.get("/api/images/1").json()["annotations"][0]["area"] == 2.0

    saved = client.post("/api/save")
    assert saved.status_code == 200
    assert saved.json()["path"].endswith("alternate.corrected.json")


def test_open_annotation_file_rejects_non_json_extension(sample_store):
    app = create_app(store=sample_store, sam2_service=FakeSAM2())
    client = TestClient(app)

    response = client.post(
        "/api/annotations/open",
        json={"file_name": "alternate.txt", "data": {"images": [], "annotations": [], "categories": []}},
    )

    assert response.status_code == 400
    assert ".json extension" in response.json()["detail"]


def test_open_annotation_file_rejects_non_coco_data_and_keeps_current_store(sample_store):
    app = create_app(store=sample_store, sam2_service=FakeSAM2())
    client = TestClient(app)

    response = client.post("/api/annotations/open", json={"file_name": "invalid.json", "data": {"images": []}})

    assert response.status_code == 400
    assert "missing required key 'annotations'" in response.json()["detail"]
    images = client.get("/api/images").json()["images"]
    assert [image["file_name"] for image in images] == ["000001.jpg", "extra.png"]


def test_open_annotation_file_rejects_unsupported_segmentation_format(sample_store):
    app = create_app(store=sample_store, sam2_service=FakeSAM2())
    client = TestClient(app)
    polygon_coco = {
        "images": [{"id": 1, "file_name": "000001.jpg", "width": 6, "height": 5}],
        "annotations": [
            {
                "id": 30,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[0, 0, 1, 0, 1, 1]],
                "area": 1.0,
                "bbox": [0.0, 0.0, 1.0, 1.0],
            }
        ],
        "categories": [{"id": 1, "name": "person_body"}],
    }

    response = client.post("/api/annotations/open", json={"file_name": "polygon.json", "data": polygon_coco})

    assert response.status_code == 400
    assert "supported COCO RLE format" in response.json()["detail"]
