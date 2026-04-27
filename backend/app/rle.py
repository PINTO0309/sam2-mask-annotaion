from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def compact_rle_to_mask(rle: dict[str, Any]) -> np.ndarray:
    decode_rle = dict(rle)
    if isinstance(decode_rle.get("counts"), str):
        decode_rle["counts"] = decode_rle["counts"].encode("ascii")
    decoded = mask_utils.decode(decode_rle)
    return decoded.astype(bool)


def mask_to_compact_rle(mask: np.ndarray) -> dict[str, Any]:
    binary = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(binary)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def mask_area(mask: np.ndarray) -> float:
    rle = mask_to_compact_rle(mask)
    decode_rle = dict(rle)
    decode_rle["counts"] = decode_rle["counts"].encode("ascii")
    return float(mask_utils.area(decode_rle))


def mask_bbox(mask: np.ndarray) -> list[float]:
    if not np.any(mask):
        return [0.0, 0.0, 0.0, 0.0]
    rle = mask_to_compact_rle(mask)
    decode_rle = dict(rle)
    decode_rle["counts"] = decode_rle["counts"].encode("ascii")
    return [float(x) for x in mask_utils.toBbox(decode_rle).tolist()]


def mask_to_png_data_url(mask: np.ndarray) -> str:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def png_data_url_to_mask(data_url: str) -> np.ndarray:
    if "," in data_url:
        _, payload = data_url.split(",", 1)
    else:
        payload = data_url
    image = Image.open(io.BytesIO(base64.b64decode(payload))).convert("L")
    return np.asarray(image) > 127
