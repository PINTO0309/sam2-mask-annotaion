from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from backend.app.ai_edges.ddn.worker import compress_edge_width
from backend.app.edge_service import EdgeDetectionService
from backend.app.main import create_app
from backend.app.rle import mask_to_compact_rle, mask_to_png_data_url, png_data_url_to_mask


class FakeEdgeService:
    def detect(self, image_path, method, low_threshold, high_threshold, ddn_thickness=1, ddn_model="m36"):
        mask = np.zeros((5, 6), dtype=bool)
        mask[2, 2:5] = True
        return {
            "edge_png": mask_to_png_data_url(mask),
            "edge_count": 3,
            "method": method,
            "requested_method": method,
            "thickness": ddn_thickness,
            "model": ddn_model,
        }


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


def test_image_edges_endpoint_returns_non_empty_edge_png(sample_store):
    image = Image.new("L", (32, 32), 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 23, 23), fill=255)
    image.save(sample_store.image_path(1))

    app = create_app(store=sample_store, sam2_service=FakeSAM2())
    client = TestClient(app)

    response = client.get("/api/images/1/edges", params={"low_threshold": 80, "high_threshold": 160})

    assert response.status_code == 200
    data = response.json()
    assert data["edge_png"].startswith("data:image/png;base64,")
    assert data["edge_count"] > 0
    assert data["method"] == "canny"
    assert np.count_nonzero(png_data_url_to_mask(data["edge_png"])) > 0


def test_image_edges_endpoint_uses_requested_ddn_method(sample_store):
    app = create_app(store=sample_store, sam2_service=FakeSAM2(), edge_service=FakeEdgeService())
    client = TestClient(app)

    response = client.get("/api/images/1/edges", params={"method": "ddn"})

    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "ddn"
    assert data["requested_method"] == "ddn"
    assert data["edge_count"] == 3
    assert data["thickness"] == 1
    assert data["model"] == "m36"


def test_image_edges_endpoint_accepts_ddn_thickness(sample_store):
    app = create_app(store=sample_store, sam2_service=FakeSAM2(), edge_service=FakeEdgeService())
    client = TestClient(app)

    response = client.get("/api/images/1/edges", params={"method": "ddn", "ddn_thickness": 2})

    assert response.status_code == 200
    assert response.json()["thickness"] == 2


def test_image_edges_endpoint_accepts_ddn_model(sample_store):
    app = create_app(store=sample_store, sam2_service=FakeSAM2(), edge_service=FakeEdgeService())
    client = TestClient(app)

    response = client.get("/api/images/1/edges", params={"method": "ddn", "ddn_model": "b36"})

    assert response.status_code == 200
    assert response.json()["model"] == "b36"


def test_image_edges_endpoint_falls_back_to_canny_when_ddn_unavailable(sample_store, tmp_path):
    app = create_app(
        store=sample_store,
        sam2_service=FakeSAM2(),
        edge_service=EdgeDetectionService(ddn_python=tmp_path / "missing-python"),
    )
    client = TestClient(app)

    response = client.get("/api/images/1/edges", params={"method": "ddn"})

    assert response.status_code == 200
    data = response.json()
    assert data["requested_method"] == "ddn"
    assert data["method"] == "canny"
    assert data["fallback"] is True
    assert "DDN Python environment not found" in data["warning"]


def test_image_edges_endpoint_rejects_invalid_thresholds(sample_store):
    app = create_app(store=sample_store, sam2_service=FakeSAM2())
    client = TestClient(app)

    assert client.get("/api/images/1/edges", params={"low_threshold": -1, "high_threshold": 160}).status_code == 400
    assert client.get("/api/images/1/edges", params={"low_threshold": 80, "high_threshold": 300}).status_code == 400
    response = client.get("/api/images/1/edges", params={"low_threshold": 160, "high_threshold": 80})
    assert response.status_code == 400
    assert "less than high_threshold" in response.json()["detail"]

    invalid_method = client.get("/api/images/1/edges", params={"method": "missing"})
    assert invalid_method.status_code == 400
    assert "method must be" in invalid_method.json()["detail"]

    invalid_thickness = client.get("/api/images/1/edges", params={"method": "ddn", "ddn_thickness": 3})
    assert invalid_thickness.status_code == 400
    assert "ddn_thickness" in invalid_thickness.json()["detail"]

    invalid_ddn_model = client.get("/api/images/1/edges", params={"method": "ddn", "ddn_model": "missing"})
    assert invalid_ddn_model.status_code == 400
    assert "ddn_model" in invalid_ddn_model.json()["detail"]


def test_image_edges_endpoint_rejects_missing_image(sample_store):
    app = create_app(store=sample_store, sam2_service=FakeSAM2())
    client = TestClient(app)

    assert client.get("/api/images/99/edges").status_code == 404


def test_ddn_edge_compression_reduces_wide_probability_band():
    edge = np.zeros((9, 9), dtype=np.float32)
    edge[2:7, 3:6] = 0.9

    one_px = compress_edge_width(edge, 1)
    two_px = compress_edge_width(edge, 2)

    assert np.count_nonzero(one_px) < np.count_nonzero(edge > 0.2)
    assert np.count_nonzero(two_px) >= np.count_nonzero(one_px)
    assert np.max(one_px) == 1
    assert np.max(two_px) == 1


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
