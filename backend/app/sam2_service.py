from __future__ import annotations

import urllib.request
import gc
from pathlib import Path

import numpy as np
from PIL import Image

from .config import SAM2_DEFAULT_MODEL_ID, SAM2_MODELS, SAM2ModelSpec


class SAM2Unavailable(RuntimeError):
    pass


class SAM2Service:
    def __init__(
        self,
        models: dict[str, SAM2ModelSpec] | None = None,
        default_model_id: str = SAM2_DEFAULT_MODEL_ID,
    ):
        self.models = models or SAM2_MODELS
        if default_model_id not in self.models:
            default_model_id = "tiny"
        self.current_model_id = default_model_id
        self._predictor = None
        self._current_image_path: Path | None = None

    def list_models(self) -> dict[str, object]:
        return {
            "current_model_id": self.current_model_id,
            "models": [
                {
                    "id": spec["id"],
                    "label": spec["label"],
                    "checkpoint_path": str(spec["checkpoint_path"]),
                    "available": spec["checkpoint_path"].exists(),
                }
                for spec in self.models.values()
            ],
        }

    def select_model(self, model_id: str) -> dict[str, object]:
        self._spec(model_id)
        if model_id != self.current_model_id:
            self.current_model_id = model_id
            self._reset_predictor()
        return self.list_models()

    def prepare_model(self, model_id: str | None = None) -> dict[str, object]:
        if model_id:
            self.select_model(model_id)
        checkpoint_path = self.ensure_checkpoint(self.current_model_id)
        return {
            "current_model_id": self.current_model_id,
            "checkpoint_path": str(checkpoint_path),
            "available": checkpoint_path.exists(),
            "models": self.list_models()["models"],
        }

    def ensure_checkpoint(self, model_id: str | None = None) -> Path:
        spec = self._spec(model_id or self.current_model_id)
        checkpoint_path = spec["checkpoint_path"]
        if checkpoint_path.exists():
            return checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_suffix(".tmp")
        urllib.request.urlretrieve(spec["checkpoint_url"], tmp_path)
        tmp_path.replace(checkpoint_path)
        return checkpoint_path

    def predict(
        self,
        image_path: Path,
        points: list[dict[str, float | int]],
        model_id: str | None = None,
    ) -> np.ndarray:
        if not points:
            raise ValueError("at least one SAM2 point prompt is required")
        if model_id:
            self.select_model(model_id)
        predictor = self._load_predictor()
        if self._current_image_path != image_path:
            image = np.asarray(Image.open(image_path).convert("RGB"))
            predictor.set_image(image)
            self._current_image_path = image_path
        coords = np.array([[float(point["x"]), float(point["y"])] for point in points], dtype=np.float32)
        labels = np.array([int(point["label"]) for point in points], dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )
        best_index = int(np.argmax(scores))
        return masks[best_index].astype(bool)

    def _spec(self, model_id: str) -> SAM2ModelSpec:
        try:
            return self.models[model_id]
        except KeyError as exc:
            raise ValueError(f"unknown SAM2 model: {model_id}") from exc

    def _reset_predictor(self) -> None:
        self._predictor = None
        self._current_image_path = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_predictor(self):
        if self._predictor is not None:
            return self._predictor
        checkpoint_path = self.ensure_checkpoint()
        spec = self._spec(self.current_model_id)
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import torch
        except Exception as exc:  # pragma: no cover - exercised only when SAM2 is installed
            raise SAM2Unavailable(
                "SAM2 is not installed. Install the optional dependency with "
                "`pip install '.[sam2]'` or `pip install sam2`."
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam2(spec["model_cfg"], str(checkpoint_path), device=device)
        self._predictor = SAM2ImagePredictor(model)
        return self._predictor
