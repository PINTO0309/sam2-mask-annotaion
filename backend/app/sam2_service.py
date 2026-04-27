from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from .config import SAM2_CHECKPOINT_PATH, SAM2_CHECKPOINT_URL, SAM2_MODEL_CFG


class SAM2Unavailable(RuntimeError):
    pass


class SAM2Service:
    def __init__(
        self,
        checkpoint_path: Path = SAM2_CHECKPOINT_PATH,
        checkpoint_url: str = SAM2_CHECKPOINT_URL,
        model_cfg: str = SAM2_MODEL_CFG,
    ):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_url = checkpoint_url
        self.model_cfg = model_cfg
        self._predictor = None
        self._current_image_path: Path | None = None

    def ensure_checkpoint(self) -> Path:
        if self.checkpoint_path.exists():
            return self.checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.checkpoint_path.with_suffix(".tmp")
        urllib.request.urlretrieve(self.checkpoint_url, tmp_path)
        tmp_path.replace(self.checkpoint_path)
        return self.checkpoint_path

    def predict(self, image_path: Path, points: list[dict[str, float | int]]) -> np.ndarray:
        if not points:
            raise ValueError("at least one SAM2 point prompt is required")
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

    def _load_predictor(self):
        if self._predictor is not None:
            return self._predictor
        self.ensure_checkpoint()
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
        model = build_sam2(self.model_cfg, str(self.checkpoint_path), device=device)
        self._predictor = SAM2ImagePredictor(model)
        return self._predictor
