from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
MODELS = {
    "tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "output": CHECKPOINT_DIR / "sam2.1_hiera_tiny.pt",
    },
    "small": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "output": CHECKPOINT_DIR / "sam2.1_hiera_small.pt",
    },
    "base_plus": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "output": CHECKPOINT_DIR / "sam2.1_hiera_base_plus.pt",
    },
    "large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "output": CHECKPOINT_DIR / "sam2.1_hiera_large.pt",
    },
}


def download(url: str, output: Path) -> None:
    if output.exists():
        print(f"checkpoint already exists: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(output)
    print(f"saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SAM2.1 checkpoints.")
    parser.add_argument("--model", choices=[*MODELS, "all"], default="tiny")
    parser.add_argument("--url", default=None, help="Override URL when downloading a single model.")
    parser.add_argument("--output", type=Path, default=None, help="Override output when downloading a single model.")
    args = parser.parse_args()
    model_ids = list(MODELS) if args.model == "all" else [args.model]
    if (args.url or args.output) and len(model_ids) != 1:
        raise SystemExit("--url and --output can only be used with a single --model value")
    for model_id in model_ids:
        spec = MODELS[model_id]
        download(args.url or spec["url"], args.output or spec["output"])


if __name__ == "__main__":
    main()
