from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[4]
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
DEFAULT_CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "ddn"
DEFAULT_ENCODER = "DDN-M36"
DEFAULT_CHECKPOINT_PATHS = {
    "DDN-S18": DEFAULT_CHECKPOINT_DIR / "ddn_s18_bsds.pth",
    "DDN-M36": DEFAULT_CHECKPOINT_DIR / "ddn_m36_bsds.pth",
    "DDN-B36": DEFAULT_CHECKPOINT_DIR / "ddn_b36_bsds.pth",
}
DEFAULT_CHECKPOINT_URL = "https://drive.google.com/uc?export=download&id=1RMIksmpAmRgccwxzzFIoZbs7203u8Q4l"
SUPPORTED_ENCODERS = tuple(DEFAULT_CHECKPOINT_PATHS.keys())


def download_with_google_drive_confirm(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if "drive.google.com" in url:
        try:
            import gdown

            result = gdown.download(url=url, output=str(destination), quiet=True, fuzzy=True)
            if result and destination.exists() and destination.stat().st_size > 0:
                return
        except Exception:
            pass

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        data = response.read()
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            destination.write_bytes(data)
            return

    text = data.decode("utf-8", errors="ignore")
    marker = "confirm="
    if marker not in text:
        raise RuntimeError("Google Drive checkpoint download requires manual confirmation")
    token = text.split(marker, 1)[1].split("&", 1)[0].split('"', 1)[0]
    confirmed_url = f"{url}&confirm={token}"
    request = urllib.request.Request(confirmed_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        destination.write_bytes(response.read())


def ensure_checkpoint(path: Path, url: str) -> Path:
    if path.exists():
        return path
    tmp_path = path.with_suffix(".tmp")
    download_with_google_drive_confirm(url, tmp_path)
    tmp_path.replace(path)
    return path


def image_to_tensor(image_path: Path, device: str):
    import torch

    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor, image.size


def normalize_tensor(tensor):
    denominator = tensor.max() - tensor.min()
    if float(denominator.detach().cpu()) <= 1e-8:
        return tensor * 0
    return (tensor - tensor.min()) / denominator


def png_data_url(edge: np.ndarray) -> str:
    edge = np.clip(edge * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(edge, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def zhang_suen_thin(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool, copy=True)
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return mask

    while True:
        changed = False
        for step in (0, 1):
            padded = np.pad(mask, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            neighbors = p2.astype(np.uint8) + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                (~p2 & p3).astype(np.uint8)
                + (~p3 & p4)
                + (~p4 & p5)
                + (~p5 & p6)
                + (~p6 & p7)
                + (~p7 & p8)
                + (~p8 & p9)
                + (~p9 & p2)
            )
            candidate = mask & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1)
            if step == 0:
                candidate &= ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                candidate &= ~(p2 & p4 & p8) & ~(p2 & p6 & p8)

            if np.any(candidate):
                mask[candidate] = False
                changed = True

        if not changed:
            return mask


def expand_to_two_pixels(mask: np.ndarray) -> np.ndarray:
    expanded = mask.astype(bool, copy=True)
    expanded[:, 1:] |= mask[:, :-1]
    expanded[1:, :] |= mask[:-1, :]
    return expanded


def compress_edge_width(edge: np.ndarray, thickness: int) -> np.ndarray:
    binary = edge > 0.2
    thinned = zhang_suen_thin(binary)
    if thickness == 2:
        thinned = expand_to_two_pixels(thinned)
    return thinned.astype(np.float32)


def run_inference(
    image_path: Path,
    checkpoint_path: Path,
    checkpoint_url: str,
    device: str,
    sample_count: int,
    thickness: int,
    encoder: str,
) -> dict[str, object]:
    sys.path.insert(0, str(VENDOR_DIR))

    import torch
    import torch.nn.functional as F
    from torch.distributions import Independent, Normal
    from model.sigma_logit_unet import Mymodel

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = ensure_checkpoint(checkpoint_path, checkpoint_url)
    args = SimpleNamespace(
        encoder=encoder,
        distribution="gs",
        cfg={"Dulbrn": 16, "ckpt": {"caformer_s18": "", "caformer_m36": "", "caformer_b36": ""}},
    )
    model = Mymodel(args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 20:
        raise RuntimeError(f"DDN checkpoint is incompatible: {len(missing)} missing keys")
    if len(unexpected) > 20:
        raise RuntimeError(f"DDN checkpoint is incompatible: {len(unexpected)} unexpected keys")

    model.eval()
    tensor, (width, height) = image_to_tensor(image_path, device)
    with torch.no_grad():
        mean, std = model(tensor)
        if sample_count > 1:
            distribution = Independent(Normal(loc=mean, scale=std + 0.001), 1)
            outputs = [distribution.rsample() for _ in range(sample_count)]
            output = torch.cat(outputs, dim=1).mean(dim=1, keepdim=True)
            output = torch.sigmoid(output)
        else:
            output = torch.sigmoid(mean)
        output = normalize_tensor(output)
        output = F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)

    edge = compress_edge_width(output.squeeze().detach().cpu().numpy(), thickness)
    return {
        "edge_png": png_data_url(edge),
        "edge_count": int(np.count_nonzero(edge > 0)),
        "method": "ddn",
        "device": device,
        "thickness": thickness,
        "encoder": encoder,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DDN-M36 edge detector worker")
    parser.add_argument("--image", required=True)
    parser.add_argument("--encoder", choices=SUPPORTED_ENCODERS, default=DEFAULT_ENCODER)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-url", default=DEFAULT_CHECKPOINT_URL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--thickness", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()

    try:
        result = run_inference(
            image_path=Path(args.image),
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT_PATHS[args.encoder],
            checkpoint_url=args.checkpoint_url,
            device=args.device,
            sample_count=max(1, args.samples),
            thickness=args.thickness,
            encoder=args.encoder,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "detail": str(exc)}), file=sys.stderr)
        return 1

    print(json.dumps({"ok": True, **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
