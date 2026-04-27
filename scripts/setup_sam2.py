from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "checkpoints" / "sam2.1_hiera_tiny.pt"


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
    parser = argparse.ArgumentParser(description="Download the default SAM2.1 tiny checkpoint.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    download(args.url, args.output)


if __name__ == "__main__":
    main()
