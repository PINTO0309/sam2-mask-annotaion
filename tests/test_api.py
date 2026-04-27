from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.rle import mask_to_png_data_url


class FakeSAM2:
    def predict(self, image_path, points):
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
    app = create_app(store=sample_store, sam2_service=FakeSAM2())
    client = TestClient(app)

    response = client.post(
        "/api/sam2/predict",
        json={"image_index": 1, "points": [{"x": 1, "y": 1, "label": 1}]},
    )

    assert response.status_code == 200
    assert response.json()["mask_png"].startswith("data:image/png;base64,")
