from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.rle import mask_to_png_data_url


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
