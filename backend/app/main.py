from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .coco_store import COCOStore
from .config import IMAGES_DIR, OUTPUT_COCO_PATH, SOURCE_COCO_PATH
from .rle import mask_to_png_data_url, png_data_url_to_mask
from .sam2_service import SAM2Service, SAM2Unavailable


class MaskPayload(BaseModel):
    mask_png: str


class CreateInstancePayload(BaseModel):
    mask_png: str | None = None


class PointPrompt(BaseModel):
    x: float
    y: float
    label: int = Field(ge=0, le=1)


class SAM2PredictPayload(BaseModel):
    image_index: int
    points: list[PointPrompt]
    model_id: str | None = None


class SAM2SelectModelPayload(BaseModel):
    model_id: str


class OpenAnnotationPayload(BaseModel):
    file_name: str
    data: dict[str, Any]


def create_app(store: COCOStore | None = None, sam2_service: SAM2Service | None = None) -> FastAPI:
    app = FastAPI(title="SAM2 COCO Mask Annotation Tool")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.store = store or COCOStore(SOURCE_COCO_PATH, IMAGES_DIR, OUTPUT_COCO_PATH)
    app.state.sam2 = sam2_service or SAM2Service()

    @app.get("/api/health")
    def health():
        return {"ok": True, "image_count": len(app.state.store.entries)}

    @app.get("/api/images")
    def list_images():
        return {"images": app.state.store.list_images()}

    @app.get("/api/images/{index}")
    def image_detail(index: int):
        try:
            return app.state.store.image_detail(index)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"image file not found: {exc}") from exc

    @app.get("/api/images/{index}/file")
    def image_file(index: int):
        try:
            path = app.state.store.image_path(index)
        except (IndexError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path)

    @app.post("/api/annotations/open")
    def open_annotation_file(payload: OpenAnnotationPayload):
        try:
            result = app.state.store.open_data(payload.data, payload.file_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, **result}

    @app.post("/api/images/{index}/instances")
    def create_instance(index: int, payload: CreateInstancePayload):
        try:
            detail = app.state.store.image_detail(index)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if payload.mask_png:
            mask = png_data_url_to_mask(payload.mask_png)
        else:
            mask = np.zeros((int(detail["height"]), int(detail["width"])), dtype=bool)
        return app.state.store.create_instance(index, mask)

    @app.put("/api/annotations/{annotation_id}/mask")
    def update_annotation_mask(annotation_id: int, payload: MaskPayload):
        try:
            mask = png_data_url_to_mask(payload.mask_png)
            return app.state.store.update_mask(annotation_id, mask)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"annotation not found: {annotation_id}") from exc

    @app.delete("/api/annotations/{annotation_id}")
    def delete_annotation(annotation_id: int):
        try:
            return app.state.store.delete_annotation(annotation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"annotation not found: {annotation_id}") from exc

    @app.get("/api/sam2/models")
    def sam2_models():
        return app.state.sam2.list_models()

    @app.post("/api/sam2/models/select")
    def sam2_select_model(payload: SAM2SelectModelPayload):
        try:
            return app.state.sam2.select_model(payload.model_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sam2/models/prepare")
    def sam2_prepare_model(payload: SAM2SelectModelPayload):
        try:
            return app.state.sam2.prepare_model(payload.model_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sam2/predict")
    def sam2_predict(payload: SAM2PredictPayload):
        try:
            path = app.state.store.image_path(payload.image_index)
            points = [point.model_dump() for point in payload.points]
            mask = app.state.sam2.predict(path, points, payload.model_id)
            return {"mask_png": mask_to_png_data_url(mask)}
        except (IndexError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SAM2Unavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/save")
    def save():
        path = app.state.store.save()
        return {"path": str(path), "download_url": "/api/download"}

    @app.post("/api/reset-edits")
    def reset_edits():
        app.state.store.reload()
        return {"ok": True, "image_count": len(app.state.store.entries)}

    @app.get("/api/download")
    def download():
        app.state.store.save()
        return FileResponse(
            app.state.store.output_path,
            media_type="application/json",
            filename=app.state.store.output_path.name,
        )

    @app.exception_handler(Exception)
    async def fallback_exception_handler(_, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app


app = create_app()
