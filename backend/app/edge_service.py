from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .config import ROOT_DIR


EdgeMethod = Literal["canny", "ddn"]
DDNModel = Literal["s18", "m36", "b36"]
DDN_MODEL_SPECS = {
    "s18": {
        "encoder": "DDN-S18",
        "checkpoint_path": ROOT_DIR / "checkpoints" / "ddn" / "ddn_s18_bsds.pth",
        "checkpoint_env": "DDN_EDGE_S18_CHECKPOINT_PATH",
        "checkpoint_url_env": "DDN_EDGE_S18_CHECKPOINT_URL",
    },
    "m36": {
        "encoder": "DDN-M36",
        "checkpoint_path": ROOT_DIR / "checkpoints" / "ddn" / "ddn_m36_bsds.pth",
        "checkpoint_env": "DDN_EDGE_M36_CHECKPOINT_PATH",
        "checkpoint_url_env": "DDN_EDGE_M36_CHECKPOINT_URL",
        "checkpoint_url": "https://drive.google.com/uc?export=download&id=1RMIksmpAmRgccwxzzFIoZbs7203u8Q4l",
    },
    "b36": {
        "encoder": "DDN-B36",
        "checkpoint_path": ROOT_DIR / "checkpoints" / "ddn" / "ddn_b36_bsds.pth",
        "checkpoint_env": "DDN_EDGE_B36_CHECKPOINT_PATH",
        "checkpoint_url_env": "DDN_EDGE_B36_CHECKPOINT_URL",
    },
}


class DDNUnavailable(RuntimeError):
    pass


def canny_edges_to_png_data_url(image_path: Path, low_threshold: int, high_threshold: int) -> tuple[str, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    ok, encoded = cv2.imencode(".png", edges)
    if not ok:
        raise ValueError("failed to encode edge image")

    payload = base64.b64encode(np.asarray(encoded).tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}", int(np.count_nonzero(edges))


class EdgeDetectionService:
    def __init__(
        self,
        ddn_python: Path | None = None,
        ddn_worker: Path | None = None,
        ddn_checkpoint: Path | None = None,
        ddn_checkpoint_url: str | None = None,
        ddn_timeout_seconds: int = 180,
    ):
        self.ddn_python = ddn_python or Path(
            os.environ.get("DDN_EDGE_PYTHON", ROOT_DIR / ".venv-ddn" / "bin" / "python")
        )
        self.ddn_worker = ddn_worker or ROOT_DIR / "backend" / "app" / "ai_edges" / "ddn" / "worker.py"
        self.ddn_checkpoint = ddn_checkpoint
        self.ddn_checkpoint_url = ddn_checkpoint_url or os.environ.get("DDN_EDGE_CHECKPOINT_URL")
        self.ddn_timeout_seconds = int(os.environ.get("DDN_EDGE_TIMEOUT_SECONDS", ddn_timeout_seconds))

    def detect(
        self,
        image_path: Path,
        method: EdgeMethod,
        low_threshold: int,
        high_threshold: int,
        ddn_thickness: int = 1,
        ddn_model: DDNModel = "m36",
    ) -> dict[str, object]:
        if method == "canny":
            edge_png, edge_count = canny_edges_to_png_data_url(image_path, low_threshold, high_threshold)
            return {"edge_png": edge_png, "edge_count": edge_count, "method": "canny", "requested_method": "canny"}

        try:
            result = self.ddn_edges(image_path, ddn_thickness, ddn_model)
            return {**result, "requested_method": "ddn"}
        except DDNUnavailable as exc:
            edge_png, edge_count = canny_edges_to_png_data_url(image_path, low_threshold, high_threshold)
            return {
                "edge_png": edge_png,
                "edge_count": edge_count,
                "method": "canny",
                "requested_method": "ddn",
                "fallback": True,
                "warning": str(exc),
            }

    def ddn_model_paths(self, model_id: DDNModel) -> tuple[str, Path, str | None]:
        spec = DDN_MODEL_SPECS[model_id]
        checkpoint = self.ddn_checkpoint
        if checkpoint is None:
            checkpoint = Path(os.environ.get(spec["checkpoint_env"], spec["checkpoint_path"]))

        checkpoint_url = self.ddn_checkpoint_url
        if checkpoint_url is None:
            checkpoint_url = os.environ.get(spec["checkpoint_url_env"], spec.get("checkpoint_url"))

        return str(spec["encoder"]), checkpoint, checkpoint_url

    def ddn_edges(self, image_path: Path, thickness: int, model_id: DDNModel) -> dict[str, object]:
        if not self.ddn_python.exists():
            raise DDNUnavailable(
                f"DDN Python environment not found: {self.ddn_python}. Run scripts/setup_ddn_edges.sh."
            )
        encoder, checkpoint, checkpoint_url = self.ddn_model_paths(model_id)
        if checkpoint_url is None and not checkpoint.exists():
            raise DDNUnavailable(
                f"{encoder} DDN checkpoint not found: {checkpoint}. Set "
                f"{DDN_MODEL_SPECS[model_id]['checkpoint_url_env']} or {DDN_MODEL_SPECS[model_id]['checkpoint_env']}."
            )
        command = [
            str(self.ddn_python),
            str(self.ddn_worker),
            "--image",
            str(image_path),
            "--encoder",
            encoder,
            "--checkpoint",
            str(checkpoint),
            "--thickness",
            str(thickness),
        ]
        if checkpoint_url:
            command.extend(["--checkpoint-url", checkpoint_url])
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT_DIR,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.ddn_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise DDNUnavailable("DDN edge detection timed out") from exc

        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "DDN worker failed"
            try:
                parsed = json.loads(detail.splitlines()[-1])
                detail = parsed.get("detail", detail)
            except (json.JSONDecodeError, IndexError):
                pass
            raise DDNUnavailable(detail)

        try:
            output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
            result = json.loads(output_lines[-1])
        except json.JSONDecodeError as exc:
            raise DDNUnavailable("DDN worker returned invalid JSON") from exc
        except IndexError as exc:
            raise DDNUnavailable("DDN worker returned no output") from exc
        if not result.get("ok"):
            raise DDNUnavailable(str(result.get("detail", "DDN worker failed")))
        return {
            "edge_png": result["edge_png"],
            "edge_count": int(result["edge_count"]),
            "method": "ddn",
            "device": result.get("device"),
            "thickness": result.get("thickness", thickness),
            "model": model_id,
            "encoder": result.get("encoder", encoder),
        }
